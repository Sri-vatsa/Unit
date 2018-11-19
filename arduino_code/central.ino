e#include <LiquidCrystal_I2C.h>

const byte numChars = 32;
char receivedChars[numChars];     // an array to store received data
bool newData = false;     // has new data been received?

int pos = 0;
char rc;    // received character

LiquidCrystal_I2C lcd(0x3F, 2, 1, 0, 4, 5, 6, 7, 3, POSITIVE);

void setup() {
  Serial.begin(115200);     //initial the Serial
  lcd.begin(16, 2);
  lcd.clear();
  lcd.print("Hello!");
}

void loop() {
  static byte ndx = 0;
  char endMarker = '\n';

  if (Serial.available() > 0) {
    rc = Serial.read();

    if (rc != endMarker) {
      receivedChars[ndx] = rc;
      ndx++;
      if (ndx >= numChars) {
        ndx = numChars - 1;
      }
    } else {
      receivedChars[ndx] = '\0';
      if (receivedChars[ndx - 2] == 'd') {
        lcd.clear();
        lcd.print("Final: ");
        float fin = pos / 12.16667;
        lcd.print(fin);
        lcd.print(" cm");
      } else if (receivedChars[ndx - 2] == 's') {
        lcd.clear();
        lcd.print("Hello!");
      } else {
        newData = true;
      }
      ndx = 0;
    }

    if (newData == true) {
      pos = atoi(receivedChars);
      float cm = pos / 12.16667;
      lcd.clear();
      lcd.print(cm);
      lcd.print(" cm");
      newData = false;
    }
  }
}
