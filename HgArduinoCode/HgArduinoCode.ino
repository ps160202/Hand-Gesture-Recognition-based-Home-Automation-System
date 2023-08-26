char serialData;

void setup() {
  pinMode(7, OUTPUT);
  pinMode(8, OUTPUT);

  Serial.begin(9600);
  Serial.setTimeout(1);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available())
  {
    serialData = Serial.read();

    if(serialData == '2') {
      while(1) {
        while(!Serial.available());
        serialData = Serial.read();

        if(serialData == '0') {
          digitalWrite(8, LOW);
        }
        else if(serialData == '1') {
          digitalWrite(8, HIGH);
        }
        else if(serialData == '2') {
          break;
        }        
      }
    }
    else if (serialData == '4') {
      while(1) {
        while(!Serial.available());
        serialData = Serial.read();

        if(serialData == '0') {
          digitalWrite(7, LOW);
        }
        else if(serialData == '1') {
          digitalWrite(7, HIGH);
        }
        else if(serialData == '4') {
          break;
        }
      }
    }
  }
}
