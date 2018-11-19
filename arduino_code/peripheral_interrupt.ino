#include <Wire.h>
#define outputA 2
#define outputB 3
#define sw 4
#define led 9

volatile int lastEncoded = 0;
volatile long encoderValue = 0;

int swState;
int swLastState = HIGH;
int toggled;  // negate side effects of long pressing switch button
int swMode;

long lastencoderValue = 0;

int lastMSB = 0;
int lastLSB = 0;

// switch debouncing
unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 50;


void setup() {
  pinMode(outputA, INPUT);  
  pinMode(outputB, INPUT);
  pinMode(sw, INPUT);
  pinMode(led, OUTPUT);

  Serial.begin (115200);

  digitalWrite(outputA, HIGH); 
  digitalWrite(outputB, HIGH); 
  digitalWrite(sw, HIGH); 
  digitalWrite(led, HIGH); // turn LED on

  //call updateEncoder() when any high/low changed seen
  //on interrupt 0 (pin 2), or interrupt 1 (pin 3)
  attachInterrupt(0, updateEncoder, CHANGE);
  attachInterrupt(1, updateEncoder, CHANGE);

  swState = HIGH; // on
  swMode = 0;
  // toggled = false;
}

void loop() {
  // save measurement if sw button is pushed
  // don't start measuring until the button is pushed again
  // ignore noise by waiting long enough by checking if you just pressed the button
  int swReading = digitalRead(sw);
  
  if (swReading != swLastState){
    lastDebounceTime = millis(); // if switch changed, reset debouncing timer
  }
  
  if ((millis() - lastDebounceTime) > debounceDelay) {
    // reading has been present for longer than debounce delay
    // it is not noise, hence can be taken as intended reading
    if (swReading != swState) {
      swState = swReading;
      // toggled = true;
      if (swState == LOW) {
        if (swMode == 0) {
          swMode = 1; // if on, turn off
          digitalWrite(led, LOW); // tape is not ready for measuring
          Serial.println('d');  // tell receiver to show final value
        } else if (swMode == 1) {
          swMode = 0;  // if off, turn on
          encoderValue = 0; // reset counter
          digitalWrite(led, HIGH); // tape is ready for measuring
          Serial.println('s');  // tell receiver to reset to 0
        }
      }
    }
  }
  // save the reading for next loop's swLastState
  swLastState = swReading;
}

void updateEncoder(){

  if (swState == 1 && swMode == 0) {
    int MSB = digitalRead(outputA); //MSB = most significant bit
    int LSB = digitalRead(outputB); //LSB = least significant bit

    int encoded = (MSB << 1) |LSB; //converting the 2 pin value to single number 
    int sum = (lastEncoded << 2) | encoded; //adding it to the previous encoded value 
    if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderValue ++; 
    if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) {
      encoderValue --; 
      if (encoderValue < 0) {
        encoderValue = 0; // reset counter to 0 if negative
      }
    }
    lastEncoded = encoded; //store this value for next time 
    Serial.println(encoderValue); // send over encoderValue
  } 
}
