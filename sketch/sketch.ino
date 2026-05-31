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

// Front HC-SR04 (body-fixed). Elegoo V4 standard; swap if us always reads 0.
const int US_TRIG = 13, US_ECHO = 12;
const unsigned long US_TIMEOUT_US = 12000;   // ~2 m cap (limits servo jitter)

ModulinoMovement imu;
ModulinoDistance tof;
ModulinoThermo   thermo;

bool has_imu=false, has_tof=false, has_thermo=false;

float g_ax=0,g_ay=0,g_az=0,g_gx=0,g_gy=0,g_gz=0;
float g_tof_mm=0, g_temp_c=0, g_us_cm=0;

int servoAngle = 90;
volatile int servoPulseUs = 1500;
unsigned long lastRefresh = 0, lastServo = 0;
const unsigned long REFRESH_MS = 50;

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

// ---- front ultrasonic (HC-SR04); cm, 0 = no echo/out of range ----
float readUltrasonicCm() {
    digitalWrite(US_TRIG, LOW);  delayMicroseconds(2);
    digitalWrite(US_TRIG, HIGH); delayMicroseconds(10);
    digitalWrite(US_TRIG, LOW);
    unsigned long dur = pulseIn(US_ECHO, HIGH, US_TIMEOUT_US);
    if (dur == 0) return 0.0;
    return dur / 58.0;                  // ~58 us per cm round-trip
}

// ---- sensors (CSV in the order their Python expects) ----
String read_sensors() {
    String s = "";
    s += String(g_ax,4); s += ","; s += String(g_ay,4); s += ","; s += String(g_az,4); s += ",";
    s += String(g_gx,2); s += ","; s += String(g_gy,2); s += ","; s += String(g_gz,2); s += ",";
    s += String(g_tof_mm,1); s += ","; s += String(g_us_cm,1); s += ","; s += String(g_temp_c,1);
    return s;
}

void refreshSensors() {
    if (has_imu && imu.update()) {
        g_ax = imu.getX();    g_ay = imu.getY();    g_az = imu.getZ();
        g_gx = imu.getRoll(); g_gy = imu.getPitch(); g_gz = imu.getYaw();
    }
    if (has_tof && tof.available()) g_tof_mm = tof.get();
    if (has_thermo) g_temp_c = thermo.getTemperature();
    g_us_cm = readUltrasonicCm();
}

void setup() {
    pinMode(PWMA,OUTPUT); pinMode(PWMB,OUTPUT);
    pinMode(AIN1,OUTPUT); pinMode(BIN1,OUTPUT); pinMode(STBY,OUTPUT);
    digitalWrite(STBY,HIGH);
    set_motors(0,0);
    pinMode(SERVO_PIN,OUTPUT);
    applyAngle(90);
    pinMode(US_TRIG,OUTPUT); pinMode(US_ECHO,INPUT);
    digitalWrite(US_TRIG,LOW);

    Modulino.begin();
    has_imu    = imu.begin();
    has_tof    = tof.begin();
    has_thermo = thermo.begin();

    Bridge.begin();
    Bridge.provide("read_sensors", read_sensors);
    Bridge.provide("set_motors", set_motors);
    Bridge.provide("set_servo", set_servo);
    Bridge.provide("get_servo", get_servo);
}

void loop() {
    Bridge.update();
    unsigned long now = millis();
    if (now - lastServo >= 20) { lastServo = now; servoPulse(); }
    if (now - lastRefresh >= REFRESH_MS) { lastRefresh = now; refreshSensors(); }
}
