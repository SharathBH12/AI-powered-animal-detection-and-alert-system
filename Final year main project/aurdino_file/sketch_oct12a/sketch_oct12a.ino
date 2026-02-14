// --- Arduino Nano Code for Blinking LED ---
int ledPin = 8;
char dataFromPC;

void setup() {
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    dataFromPC = Serial.read();

    if (dataFromPC == '1') {
      // Blink LED for 3 seconds
      unsigned long startTime = millis();
      while (millis() - startTime < 1000) {  // 3 seconds
        digitalWrite(ledPin, HIGH);
        delay(300);
        digitalWrite(ledPin, LOW);
        delay(300);
      }
    } 
    else if (dataFromPC == '0') {
      digitalWrite(ledPin, LOW); // Turn off if signal 0 received
    }
  }
}
