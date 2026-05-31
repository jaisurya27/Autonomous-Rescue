// SPDX-License-Identifier: MPL-2.0
//
// Recon Rover - MCU firmware (YOUR hardware + their Python's sensor interface)
// -----------------------------------------------------------------------------
// Bridge interface expected by the Python brain:
//   read_sensors() -> CSV: ax,ay,az,gx,gy,gz,tof_mm,us_cm,temp_c
//                     us_cm = front HC-SR04 (body-fixed, always forward).
//                     tof_mm = Modulino ToF on the SERVO HEAD (pans).
//   set_motors(left,right) -> signed PWM, your TB6612 driver
//   set_servo(deg)         -> pan servo (library-free pulse)
//   get_servo()            -> current servo angle
//
// YOUR confirmed pins: PWMA=5 PWMB=6 AIN1=7 BIN1=8 STBY=3 SERVO=11
//   Front HC-SR04 (Elegoo V4): TRIG=13, ECHO=12. If us reads 0 always, swap them.
// Movement gyro is on getPitch() (your vertical mount). Modulino on Wire1.

#include <Arduino_RouterBridge.h>
#include <Arduino_Modulino.h>

const int PWMA = 5, PWMB = 6, AIN1 = 7, BIN1 = 8, STBY = 3;
const int LEFT_FWD = HIGH, RIGHT_FWD = HIGH;
const int SERVO_PIN = 11, SERVO_MIN = 20, SERVO_MAX = 160;

const int RED_PIN   = A0;
const int GREEN_PIN = A1;
const int BLUE_PIN  = A2;
const int BUZZER    = 10;

// HC-SR04 removed — us_cm field in CSV is always 0.0

ModulinoMovement imu;
ModulinoDistance tof;
ModulinoThermo   thermo;

bool has_imu=false, has_tof=false, has_thermo=false;

float g_ax=0,g_ay=0,g_az=0,g_gx=0,g_gy=0,g_gz=0;
float g_tof_mm=0, g_temp_c=0;

int servoAngle = 90;
volatile int servoPulseUs = 1500;
unsigned long lastRefresh = 0, lastServo = 0, lastLED = 0;
const unsigned long REFRESH_MS = 50;

// ---- indicator state ----
// 0=off  1=explore(solid blue)  2=sweep(blink red+beep)
// 3=return(blink green)  4=person(one beep, then restore)
volatile int  g_indicator = 0;
int  g_prev_indicator = 0;   // mode before a person beep, to restore after
bool g_led_on = false;
unsigned long g_beep_until = 0;
int  g_beep_phase = 0;       // 0=first beep 1=gap 2=second beep 3=silence (sweep only)

void set_rgb(bool r, bool g, bool b) {
    digitalWrite(RED_PIN,   r ? HIGH : LOW);
    digitalWrite(GREEN_PIN, g ? HIGH : LOW);
    digitalWrite(BLUE_PIN,  b ? HIGH : LOW);
}
void set_indicator(int mode) {
    if (mode == 4) {           // person: one short beep then restore
        g_prev_indicator = g_indicator;
        g_indicator = 4;
        g_beep_until = millis() + 120;
        tone(BUZZER, 1200);
    } else {
        g_indicator = mode;
        noTone(BUZZER);
    }
    g_led_on = false;
    g_beep_until = 0;
    // mode 2 beep-beep: start at phase 3 so first updateIndicator tick fires phase 0
    // (first beep). Initialising at 0 skips phase 0 and starts with silence instead.
    g_beep_phase = (mode == 2) ? 3 : 0;
}

// ---- motors (your TB6612: one dir pin per side) ----
void driveOne(int pwmPin, int dirPin, int fwd, int speed) {
    if (speed < 0) { digitalWrite(dirPin, fwd == HIGH ? LOW : HIGH); speed = -speed; }
    else           { digitalWrite(dirPin, fwd); }
    if (speed > 255) speed = 255;
    analogWrite(pwmPin, speed);
}
void set_motors(int left, int right) {
    if (left  > 255) left = 255;  if (left  < -255) left = -255;
    if (right > 255) right = 255; if (right < -255) right = -255;
    driveOne(PWMA, AIN1, LEFT_FWD,  left);
    driveOne(PWMB, BIN1, RIGHT_FWD, right);
}

// ---- servo (library-free) ----
void applyAngle(int deg) {
    if (deg < SERVO_MIN) deg = SERVO_MIN;
    if (deg > SERVO_MAX) deg = SERVO_MAX;
    servoAngle = deg;
    servoPulseUs = 500 + (long)deg * 2000 / 180;
}
void set_servo(int deg) { applyAngle(deg); }
int  get_servo()        { return servoAngle; }
void servoPulse() {
    digitalWrite(SERVO_PIN, HIGH);
    delayMicroseconds(servoPulseUs);
    digitalWrite(SERVO_PIN, LOW);
}

// ---- tunes (non-blocking, driven from loop) ----
// Tune IDs: 1 = happy birthday (~10s), 2 = fade-out beeps (~5s)
// Each entry: {freq_hz, duration_ms}. 0 freq = rest.
struct Note { uint16_t freq; uint16_t dur; };

static const Note TUNE_BIRTHDAY[] PROGMEM = {
    {264,200},{264,200},{0,50},{296,300},{264,300},{352,300},{330,600},{0,150},
    {264,200},{264,200},{0,50},{296,300},{264,300},{396,300},{352,600},{0,150},
    {264,200},{264,200},{0,50},{528,300},{440,300},{352,300},{330,300},{296,600},{0,150},
    {470,200},{470,200},{0,50},{440,300},{352,300},{396,300},{352,600},{0,300}
};
static const Note TUNE_FADEOUT[] PROGMEM = {
    {880,120},{0,60},{780,110},{0,60},{680,100},{0,60},
    {580,90}, {0,60},{480,80}, {0,60},{380,70}, {0,60},
    {280,60}, {0,60},{200,50}, {0,80},{140,40}, {0,100},
    {100,30}, {0,200}
};
static const uint8_t TUNE_BIRTHDAY_LEN = sizeof(TUNE_BIRTHDAY)/sizeof(Note);
static const uint8_t TUNE_FADEOUT_LEN  = sizeof(TUNE_FADEOUT)/sizeof(Note);

int            g_tune_id   = 0;   // 0=off, 1=birthday, 2=fadeout
uint8_t        g_tune_note = 0;
unsigned long  g_tune_until = 0;

void play_tune(int id) {
    g_tune_id   = id;
    g_tune_note = 0;
    g_tune_until = 0;
    noTone(BUZZER);
    // reset indicator beep so they don't fight
    g_beep_until = 0;
}

void updateTune() {
    if (g_tune_id == 0) return;
    unsigned long now = millis();
    if (now < g_tune_until) return;   // still playing current note

    const Note* tune;
    uint8_t len;
    if (g_tune_id == 1) { tune = TUNE_BIRTHDAY; len = TUNE_BIRTHDAY_LEN; }
    else                { tune = TUNE_FADEOUT;  len = TUNE_FADEOUT_LEN;  }

    if (g_tune_note >= len) {
        g_tune_id = 0; noTone(BUZZER); return;
    }
    Note n;
    memcpy_P(&n, &tune[g_tune_note], sizeof(Note));
    g_tune_note++;
    g_tune_until = now + n.dur;
    if (n.freq > 0) tone(BUZZER, n.freq);
    else            noTone(BUZZER);
}

// ---- indicator update (call every loop, non-blocking) ----
void updateIndicator() {
    unsigned long now = millis();

    // mode 4: person beep — one short tone, then restore previous mode
    if (g_indicator == 4) {
        if (now >= g_beep_until) {
            noTone(BUZZER);
            g_indicator = g_prev_indicator;
            g_led_on = false;
            g_beep_until = 0;
            g_beep_phase = (g_prev_indicator == 2) ? 3 : 0;
        }
        return;   // LED stays as-is during the beep
    }

    if (g_indicator == 0) { set_rgb(0,0,0); noTone(BUZZER); return; }
    if (g_indicator == 1) { set_rgb(0,0,1); noTone(BUZZER); return; }  // solid blue

    // modes 2 (sweep/red blink + beep-beep) and 3 (return/green blink)
    const unsigned long BLINK_HALF = 300;   // ms per half-cycle
    // sweep beep pattern: beep(150ms) gap(100ms) beep(150ms) silence(500ms)
    const unsigned long BEEP_DURATIONS[] = {150, 100, 150, 500};

    if (now - lastLED >= BLINK_HALF) {
        lastLED = now;
        g_led_on = !g_led_on;
        if (g_indicator == 2) set_rgb(g_led_on, 0, 0);   // blink red
        else                  set_rgb(0, g_led_on, 0);   // blink green
    }

    if (g_indicator == 2) {
        // beep-beep pattern driven by g_beep_phase
        if (now >= g_beep_until) {
            g_beep_phase = (g_beep_phase + 1) % 4;
            g_beep_until = now + BEEP_DURATIONS[g_beep_phase];
            if (g_beep_phase == 0 || g_beep_phase == 2) tone(BUZZER, 900);
            else                                         noTone(BUZZER);
        }
    } else {
        noTone(BUZZER);
    }
}

// ---- sensors (CSV in the order their Python expects) ----
String read_sensors() {
    String s = "";
    s += String(g_ax,4); s += ","; s += String(g_ay,4); s += ","; s += String(g_az,4); s += ",";
    s += String(g_gx,2); s += ","; s += String(g_gy,2); s += ","; s += String(g_gz,2); s += ",";
    s += String(g_tof_mm,1); s += ","; s += String(0.0,1); s += ","; s += String(g_temp_c,1);
    return s;
}

void refreshSensors() {
    if (has_imu && imu.update()) {
        g_ax = imu.getX();    g_ay = imu.getY();    g_az = imu.getZ();
        g_gx = imu.getRoll(); g_gy = imu.getPitch(); g_gz = imu.getYaw();
    }
    if (has_tof && tof.available()) g_tof_mm = tof.get();
    if (has_thermo) g_temp_c = thermo.getTemperature();
    // US is read separately in loop() on its own timer to avoid
    // pulseIn blocking during a Bridge call (causes heap corruption).
}

void setup() {
    pinMode(PWMA,OUTPUT); pinMode(PWMB,OUTPUT);
    pinMode(AIN1,OUTPUT); pinMode(BIN1,OUTPUT); pinMode(STBY,OUTPUT);
    digitalWrite(STBY,HIGH);
    set_motors(0,0);
    pinMode(SERVO_PIN,OUTPUT);
    applyAngle(90);
    pinMode(RED_PIN,OUTPUT); pinMode(GREEN_PIN,OUTPUT); pinMode(BLUE_PIN,OUTPUT);
    pinMode(BUZZER,OUTPUT);
    set_rgb(0,0,0); noTone(BUZZER);

    Modulino.begin();
    has_imu    = imu.begin();
    has_tof    = tof.begin();
    has_thermo = thermo.begin();

    Bridge.begin();
    Bridge.provide("read_sensors",  read_sensors);
    Bridge.provide("set_motors",    set_motors);
    Bridge.provide("set_servo",     set_servo);
    Bridge.provide("get_servo",     get_servo);
    Bridge.provide("set_indicator", set_indicator);
    Bridge.provide("play_tune",     play_tune);
}

void loop() {
    unsigned long now = millis();
    if (now - lastServo >= 20)           { lastServo = now;   servoPulse(); }
    if (now - lastRefresh >= REFRESH_MS) { lastRefresh = now; refreshSensors(); }
    updateTune();
    if (g_tune_id == 0) updateIndicator();  // don't let indicator beeps fight the tune
    Bridge.update();
}
