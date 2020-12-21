#include <SoftwareSerial.h>

#include <ESP8266WiFi.h>
#include <WiFiClient.h>

// TX<->RX, RX<->TX
//SoftwareSerial softSerial(G12, G14); // RX, TX
SoftwareSerial softSerial(D2, D3); // RX, TX
                                   // Green, Violet

const char* ssid      = "scalar";
const char* password  = "20190305";

const char* host      = "192.168.0.16";
const uint16_t port   = 55555;

WiFiClient client;

String readAll() {
  String msg = "";
  while (client.available() > 0) {
    char _data = client.read();
    msg += _data;
    
    delay(50);
  }
  msg += '\r';
  msg += '\n';

  return msg;
}

void setup() {
  Serial.begin(9600);
  softSerial.begin(9600);


  // Connect to WIFI AP.
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  while(WiFi.status() != WL_CONNECTED){
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("Wifi connected!");


  // Connect to TCP Socket server.
  Serial.print("Connecting to ");
  Serial.println(host);
  while (!client.connect(host, port)) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("TCP Socket connected!");
}

void loop() {
  // Receive data from server. 
  if (client.available() > 0) {
    String msg = readAll();
    
    Serial.print("REQUEST: ");
    Serial.println(msg);

    char* ptr;
    int _data = strtol(msg.c_str(), &ptr, 2);
    softSerial.write(_data);
  } 

  // Send serial input to server.
  /* if (Serial.available() > 0) {
    client.write(Serial.read());
  } */
  if (softSerial.available() > 0) {
    char data = Serial.read();
    Serial.write(data);
    //client.write(data);
  } /**/
}
