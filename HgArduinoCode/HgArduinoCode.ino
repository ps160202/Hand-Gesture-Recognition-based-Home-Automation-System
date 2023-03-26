int flag = 0;
char serialData;

void setup() {
  pinMode(13, OUTPUT);

  Serial.begin(9600);
  Serial.setTimeout(1);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available())
  {
    serialData = Serial.read();

    if(serialData == '1') {
      digitalWrite(13, LOW);
    }
    else if (serialData == '0') {
      digitalWrite(13, HIGH);
    }
  }
}
