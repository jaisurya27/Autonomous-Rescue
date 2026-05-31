/*
 * sensor_bridge.ino — STM32 firmware, BRIDGE version
 * Fix: don't let sensor reads (esp. ultrasonic pulseIn) starve Bridge.update().
 *      Throttle sensor refresh; service the bridge every loop.
 */

#include <Modulino.h>
#include <Wire.h>

ModulinoMovement imu;
ModulinoDistance tof;
ModulinoThermo  thermo;

#define ENA 5
#define IN1 7
#define IN2 8
#define ENB 6
#define IN3 9
#define IN4 10
#define TRIG_PIN 12
#define ECHO_PIN 11

float g_ax=0,g_ay=0,g_az=0,g_gx=0,g_gy=0,g_gz=0;
float g_tof_mm=0, g_us_cm=0, g_temp_c=0;

bool has_imu=false, has_tof=false, has_thermo=false;
unsigned long lastRefresh=0;
const unsigned long REFRESH_MS=50;   // read sensors ~20Hz, not every loop

String read_sensors() {
  String s = "";
  s += String(g_ax,4); s += ","; s += String(g_ay,4); s += ","; s += String(g_az,4); s += ",";
  s += String(g_gx,2); s += ","; s += String(g_gy,2); s += ","; s += String(g_gz,2); s += ",";
  s += String(g_tof_mm,1); s += ","; s += String(g_us_cm,1); s += ","; s += String(g_temp_c,1);
  return s;
}

void set_motors(int left, int right) {
  left  = constrain(left,  -255, 255);
  right = constrain(right, -255, 255);
  if (left > 0)      { digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);  analogWrite(ENA, left); }
  else if (left < 0) { digitalWrite(IN1,LOW);  digitalWrite(IN2,HIGH); analogWrite(ENA, -left); }
  else               { digitalWrite(IN1,LOW);  digitalWrite(IN2,LOW);  analogWrite(ENA, 0); }
  if (right > 0)      { digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);  analogWrite(ENB, right); }
  else if (right < 0) { digitalWrite(IN3,LOW);  digitalWrite(IN4,HIGH); analogWrite(ENB, -right); }
  else                { digitalWrite(IN3,LOW);  digitalWrite(IN4,LOW);  analogWrite(ENB, 0); }
}

float readUltrasonic() {
  digitalWrite(TRIG_PIN, LOW);  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long dur = pulseIn(ECHO_PIN, HIGH, 15000);  // shorter timeout (~2.5m) so we don't stall
  if (dur == 0) return 400.0;
  return dur * 0.01715;
}

void refreshSensors() {
  if (has_imu && imu.update()) {
    g_ax = imu.getX();    g_ay = imu.getY();    g_az = imu.getZ();
    g_gx = imu.getRoll(); g_gy = imu.getPitch(); g_gz = imu.getYaw();
  }
  if (has_tof && tof.available()) g_tof_mm = tof.get();
  g_us_cm = readUltrasonic();
  if (has_thermo) g_temp_c = thermo.getTemperature();
}

void setup() {
  Modulino.begin();
  has_imu    = imu.begin();
  has_tof    = tof.begin();
  has_thermo = thermo.begin();

  pinMode(ENA,OUTPUT); pinMode(IN1,OUTPUT); pinMode(IN2,OUTPUT);
  pinMode(ENB,OUTPUT); pinMode(IN3,OUTPUT); pinMode(IN4,OUTPUT);
  pinMode(TRIG_PIN,OUTPUT); pinMode(ECHO_PIN,INPUT);
  set_motors(0,0);

  Bridge.begin();
  Bridge.provide("read_sensors", read_sensors);
  Bridge.provide("set_motors", set_motors);
}

void loop() {
  Bridge.update();                       // ALWAYS service the bridge first, every loop
  unsigned long now = millis();
  if (now - lastRefresh >= REFRESH_MS) { // throttle the (slower) sensor reads
    lastRefresh = now;
    refreshSensors();
  }
}
