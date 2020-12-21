// NOTE: Finger Release Tension (in ms) 
const int tension_constant[5] = {
  200,  // Thumb 
  50,  // Index Finger (2nd)
  50,  // 3rd Finger
  50,  // 4th Finger
  50}; // 5th Finger

// EMG_FILTER VALUE: min = 0 (no average), max = 90 (aggressive average)
const int EMG_FILTER = 10;
const int GRADUAL_POWER_INIT = 25;
const int GRADUAL_POWER_INCREASE = 1;
const int GRADUAL_POWER_MAX = 90;
const int GRADUAL_GRASPING_TIME_MIN = 200;
const int periodMili = 90;

int gradual_grasping_time = 0;

uint8_t program_mode = 0;

const int total_grasp_period = 10;
const int on_grasp_period = 10;

const int total_release_period = 10;
const int off_release_period = 5;
const int on_release_period = 5;

const int total_powerdown_period = 15;

const int emg_high = 80;
const int emg_low = 20;
const int emg_diff = 100;
int emg_peak = 0;
int emg_valley = 0;

const int low_limit_voltage = 108;

#include <MemoryFree.h>
#include <Wire.h>
#include <SSD1306Ascii.h>
#include <SSD1306AsciiWire.h>
#include <avr/wdt.h>

#define I2C_ADDRESS 0x3C

int powerSwitchPin = 15;
int trEnablePin = 14;
int motorPin[6] = {4, 5, 6, 7, 8, 9};
int switchPin[3] = {16, 10, A1};
int VOLTAGE_PIN = A2;
int sensorPin[2] = {A0, A3};
int switch_status[5] = {0,};
int switch_value[2] = {0,};
long steady_state = 0;

int loop_count = 0;
int low_voltage_loop = 0;

int getInitialSteadyState();
void grasp_movement(int finger_data);
void release_movement(int finger_data);
int getVoltage10x();

void setup() {
  CLKPR = 0x80;
  CLKPR = 0;
  
  pinMode(powerSwitchPin, INPUT_PULLUP);
  pinMode(trEnablePin, OUTPUT);
  digitalWrite(trEnablePin, HIGH);
  
  Wire.begin();
  Wire.setClock(400000L);
  delay(500);
  
  for (int i=0; i<6; ++i) {
    pinMode(motorPin[i], OUTPUT);
    digitalWrite(motorPin[i], LOW);
  }

  Serial.begin(9600);
  Serial1.begin(9600);
  delay(1000);
  Serial.println("Hi there!");

  pinMode(switchPin[0], INPUT_PULLUP);
  
  //getInitialSteadyState();

  // DC MOTOR MOVEMENT TEST
  Serial.println("[GRASP]");
  grasp_movement(0b11111);
  delay(500);

  // DC MOTOR RELEASE MOVEMENT
  Serial.println("[RELEASE]");
  release_movement(0b11111);
}

int getVoltage10x() {
  int voltage = 0;
  analogRead(VOLTAGE_PIN);
  analogRead(VOLTAGE_PIN);
  analogRead(VOLTAGE_PIN);
  analogRead(VOLTAGE_PIN);
  for (int i=0; i<8; ++i) {
    voltage += analogRead(VOLTAGE_PIN);
  }
  voltage = ((voltage / 8) * 3 * 5.1 / 102);
  return voltage;
}

// Read swichPin[1,2] and map to switch_status[1,2] and [3,4].
void readSwitchStatus() {
  int analog_value;  // Average of switchPin
  int i;
  switch_status[0] = digitalRead(switchPin[0]);
  analogRead(switchPin[1]);
  delayMicroseconds(60);
  for (analog_value=i=0; i<8; ++i) {
    analog_value += analogRead(switchPin[1]);
    delayMicroseconds(60);
  }
  analog_value /= 8;
  switch_value[0] = analog_value;
  //Serial.println(analog_value);
  if (analog_value > 850) {
    switch_status[1] = 1;
    switch_status[2] = 1;
  } else if (analog_value > 300 && analog_value < 380) {
    switch_status[1] = 1;
    switch_status[2] = 0;
  } else if (analog_value > 450 && analog_value < 550) {
    switch_status[1] = 0;
    switch_status[2] = 1;
  } else if (analog_value > 220 && analog_value < 300) {
    switch_status[1] = 0;
    switch_status[2] = 0;
  } else {
    switch_status[1] = 0;
    switch_status[2] = 0;
  }
  analogRead(switchPin[2]);
  delayMicroseconds(60);
  for (analog_value=i=0; i<8; ++i) {
    analog_value += analogRead(switchPin[2]);
    delayMicroseconds(60);
  }
  analog_value /= 8;
  switch_value[1] = analog_value;
  if (analog_value > 850) {
    switch_status[4] = 1;
    switch_status[3] = 1;
  } else if (analog_value > 300 && analog_value < 380) {
    switch_status[4] = 0;
    switch_status[3] = 1;
  } else if (analog_value > 450 && analog_value < 550) {
    switch_status[4] = 1;
    switch_status[3] = 0;
  } else if (analog_value > 220 && analog_value < 300) {
    switch_status[4] = 0;
    switch_status[3] = 0;
  } else {
    switch_status[4] = 0;
    switch_status[3] = 0;
  }
}

int finger_state = 0b11111;
void loop() {
  if (Serial1.available() > 0) {
    char data = Serial1.read();
    //data = -1;

    if (0 <= data && data < 32) {  // filter
      Serial.print("REQUEST: ");
      Serial.println((int)data);
      //Serial1.write(data);  // RESPONSE BACK

      int grasp_bit = (finger_state ^ data) & finger_state;
      int release_bit = (finger_state ^ data) & data;

      release_movement(release_bit);
      grasp_movement(grasp_bit);
      
      finger_state = data;
    } else {
      Serial.print("TRASH: ");
      Serial.println((int)data);
    }
  }
  
  /*
  if (Serial1.available() > 0) {
    //Serial.println("RECV");
    int data = Serial1.read();
    Serial.write((char)data);
  }
  if (Serial.available() > 0) {
    Serial1.write((char)Serial.read());
  } */
}

// Read steady value from sensorPin[0] 100 times and save in `steady_state`.
int getInitialSteadyState() {
  steady_state = 0;
  analogRead(sensorPin[0]);
  analogRead(sensorPin[0]);
  for (int i=0; i<100; ++i) {
    steady_state += analogRead(sensorPin[0]);
    delay(1);
  }
}

void grasp_movement(int finger_data) {
  int finger[5];
  int tension[5];
  int cur_voltage;
  int idl_voltage;
  int f_power_saturation[5] = {0, };
  int f_power[5] = {10, 10, 10, 10, 10};
  int f_power_normalized[5] = {10, 10, 10, 10, 10}; 
  int f_timer[5] = {0, 0, 0, 0, 0};
  int f_on[5] = {0, 0, 0, 0, 0};
  int order[5] = {1, 2, 0, 3, 4};
  int flag[5] = {0,};   // 0: MOVE, 1: TENSION, 2: STOP, 3: NONE(EXIT)
  int timeout[5] = {0,};
  int focused_power = 0;
  const int low_power = 3;
  const int limit_voltage = 12;
  const int low_voltage = 106;
  char voltage_buf[12]= {0,};
  int balance_power_flag = 0;
  int loop_iteration = 0;
  uint16_t timer_count = 0;
  uint16_t sleep_time = 0;
  
  for (int i = 0; i < 5; ++i) {
    finger[i] = (finger_data & (1 << i)) != 0;
  }

  digitalWrite(motorPin[0], HIGH);
  for (int i=0; i<5; ++i) {
    if (finger[i] == 0) {
      flag[i] = 3;
    } else {
      flag[i] = 0;
    }
    f_power[i] = on_grasp_period;
    digitalWrite(motorPin[i+1], HIGH);
    timeout[i] = 650;
    tension[i] = 5+5*i;
  }
  
  cur_voltage = idl_voltage = getVoltage10x();

  int cumulative_delay = 0;
  for (int i=0; i<5; ++i) {
    if (finger[order[i]] == 0) continue;
    timeout[order[i]] += cumulative_delay;
    digitalWrite(motorPin[order[i]+1], LOW);
    f_on[order[i]] = 1;
    delay(130);
    cumulative_delay += 130;
  }

  while(1) {
    timer_count = micros();
    loop_iteration++;
    //Serial.println(loop_iteration);
    cur_voltage = getVoltage10x();
    
    // IF LOW VOLTAGE DETECTED (1.0 volt lower)
    if (idl_voltage - cur_voltage > limit_voltage || cur_voltage < low_voltage) {
      balance_power_flag = 1;
    }
    if (balance_power_flag) { // IF VOLTAGE DROP OCCURS
      balance_power_flag = 0;
      focused_power = 0;
      for (int i=0; i<5; ++i) {
        f_power[order[i]] = 0;
        if (finger[order[i]] != 0 && flag[order[i]] < 2) { // when i-th finger is active 
          if (focused_power == 0) {
            if (cur_voltage < low_voltage) {
              f_power[order[i]] = on_grasp_period - 2;
              focused_power += on_grasp_period - 2;
            } else if (cur_voltage < low_voltage + 3) {
              f_power[order[i]] = on_grasp_period - 1;
              focused_power += on_grasp_period - 1;
            } else {
              f_power[order[i]] = on_grasp_period;
              focused_power += on_grasp_period;
            }
          } else {
            if (cur_voltage < low_voltage) {
              f_power[order[i]] = 2;
            } else if (cur_voltage < low_voltage + 3) {
              f_power[order[i]] = 2;
            } else if (cur_voltage < low_voltage + 6) {
              f_power[order[i]] = 3;
            } else {
              f_power[order[i]] = 4;
            }
          }
        }
      }
    }

    for (int i=0; i<5; ++i) {
      if (finger[i] != 0 && flag[i] < 2) { // when i-th finger is active 
        if (f_timer[i] >= f_power_normalized[i]) { // when it meets to the end of on period, then turn it off
          if (f_power_normalized[i] != total_grasp_period) { // only when on period is different from the total period
            digitalWrite(motorPin[i+1], HIGH); // high is off when grasping
            f_on[i] = 0;
          }
        }
        if (f_timer[i] >= total_grasp_period) { // when it meets to the end of total period, just turn it on
          f_timer[i] = 0;
          if (f_power_normalized[i] > 0) {
            digitalWrite(motorPin[i+1], LOW); // low is on when grasping 
            f_on[i] = 1;
          }
        }
        f_timer[i]++;
      }
      else if (finger[i] != 0 && flag[i] == 2) { // when i-th finger is in power-down mode 
        if (f_timer[i] >= f_power_normalized[i]) { // when it meets to the end of on period, then turn it off
          if (f_power_normalized[i] != total_powerdown_period) { // only when on period is different from the total period
            digitalWrite(motorPin[i+1], HIGH); // high is off when grasping
            f_on[i] = 0;
            f_power[i]--;
            if (f_power[i] <= 0) {
              flag[i] = 3; // POWER DOWN (END OF RELEASE OF I-th Finger)
              balance_power_flag = 1;
            }
          }
        }
        if (f_timer[i] >= total_powerdown_period) { // when it meets to the end of total period, just turn it on
          f_timer[i] = 0;
          if (f_power_normalized[i] > 0) {
            digitalWrite(motorPin[i+1], LOW); // low is on when grasping 
            f_on[i] = 1;
          }
        }
        f_timer[i]++;
      }
    }
    
    if (loop_iteration % 20 == 0) {
      readSwitchStatus();
    }
    
    for (int i=0; i<5; ++i) {
      if (f_on[i]) { // when finger motor is on
        timeout[i] -= 1;
        if (flag[i] == 1) {
          tension[i] -= 1;
          f_power_saturation[i]++;
          if (f_power_normalized[i] > 5 && f_power_saturation[i] > 8) {
            if (f_power[i] > 0) {
              f_power[i]--;
            }
            f_power_saturation[i] = 0;
          }
        }
      }
      if (finger[i] != 0 && flag[i] < 2) { // when i-th finger is active 
        if ((timeout[i] <= 450) && (switch_status[i] == 0)) {
          flag[i] = 1;
        }
        if (timeout[i] < 0) {
          flag[i] = 1;
        }
        if (tension[i] < 0) {
          flag[i] = 2;
          digitalWrite(motorPin[i+1], HIGH); // high is off when grasping 
          f_power[i] = total_powerdown_period * 4 / 5;
          f_timer[i] = 0;
        }
      }
    }
    if ((flag[0] == 3) && (flag[1] == 3) && (flag[2] == 3) && (flag[3] == 3) && (flag[4] == 3)) {
      break;
    }
    sleep_time = micros() - timer_count;
    if (sleep_time < 1000) {
      delayMicroseconds(1000 - sleep_time);
    }
  }

  digitalWrite(motorPin[0], HIGH);
  for (int i=0; i<5; ++i) {
    digitalWrite(motorPin[i+1], HIGH); 
    flag[i] = 0;
    timeout[i] = 700;
  }
  emg_peak = 0;
  emg_valley = 400;
}


void release_movement(int finger_data) {
  int finger[5];
  int tension[5];
  int f_power[5] = {10, 10, 10, 10, 10};
  int f_timer[5] = {0, 0, 0, 0, 0};
  int f_on[5] = {0, 0, 0, 0, 0};
  int flag[5] = {0,};
  int timeout[5] = {0,};
  char voltage_buf[12]= {0,};
  int cur_voltage = 0;
  int loop_iteration = 0;

  for (int i = 0; i < 5; ++i) {
    finger[i] = (finger_data & (1 << i)) != 0;
  }
  

  cur_voltage = getVoltage10x();

  digitalWrite(motorPin[0], LOW);
  
  for (int i=0; i<5; ++i) {
    if (finger[i] == 0) {
      flag[i] = 2;
    } else {
      flag[i] = 0;
    }
    f_power[i] = on_release_period;
    digitalWrite(motorPin[i+1], LOW);
    timeout[i] = 600;
    tension[i] = tension_constant[i];
  }
  timeout[0] = 400;
  
  digitalWrite(motorPin[0], LOW);
  for (int i=0; i<5; ++i) {
    if (finger[i] == 0) continue;
    digitalWrite(motorPin[i+1], HIGH);
    delay(20);
    timeout[i] -= 20*(4-i);
  }
  
  while(1) {
    loop_iteration++;
    for (int i=0; i<5; ++i) {
      if (finger[i] != 0 && flag[i] < 2) { // when i-th finger is active 
        if (f_timer[i] >= f_power[i]) { // when it meets to the end of on period, then turn it off
          if (f_power[i] != total_release_period) { // only when on period is different from the total period
            digitalWrite(motorPin[i+1], LOW); // low is off when releasing 
            f_on[i] = 0;
          }
        }
        if (f_timer[i] >= total_release_period) { // when it meets to the end of total period, just turn it on
          f_timer[i] = 0;
          if (f_power[i] > 0) {
            digitalWrite(motorPin[i+1], HIGH); // high is on when releasing
            f_on[i] = 1;
          }
        }
        f_timer[i]++;
      }
    }

    if (loop_iteration % 5 == 0) {
      readSwitchStatus();
    }
    
    for (int i=0; i<5; ++i) {
      if (f_on[i]) { // when finger motor is on
        timeout[i] -= 2;
        if (flag[i] == 1) {
          tension[i] -= 2;
        }
      }
      if (finger[i] != 0 && flag[i] < 2) { // when i-th finger is active 
        if ((timeout[i] <= 300) && (switch_status[i] == 0)) {
          flag[i] = 1;
        }
        if (timeout[i] < 0) {
          flag[i] = 1;
        }
        if (tension[i] < 0) {
          flag[i] = 2;
          digitalWrite(motorPin[i+1], LOW); // low is off when releasing 
        }
      }
    }
    if ((flag[0] == 2) && (flag[1] == 2) && (flag[2] == 2) && (flag[3] == 2) && (flag[4] == 2)) {
      break;
    }
    delayMicroseconds(1000);
  }

  digitalWrite(motorPin[0], LOW);
  for (int i=0; i<5; ++i) {
    digitalWrite(motorPin[i+1], LOW);
    flag[i] = 0;
    timeout[i] = 1000;
  }
  emg_peak = 0;
  emg_valley = 400;
}
