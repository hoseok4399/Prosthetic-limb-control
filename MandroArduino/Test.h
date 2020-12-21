#ifndef Test_h
#define Test_h

char str[1024];
void _printf(int pinnum, int flagval) {
  return;
  int m = millis();
  sprintf(str, "%d %d %d", m, pinnum, flagval);
  Serial.println(str);
}

#endif
