#define ENABLE_USER_AUTH
#define ENABLE_DATABASE

#include <SPI.h>
#include <MFRC522.h>
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <FirebaseClient.h>
#include "time.h"

// Network and Firebase credentials
#define WIFI_SSID "Galaxy A04s95E1"
#define WIFI_PASSWORD "babyboogal00"

#define Web_API_KEY "AIzaSyA-8_tGoa1Mm5HW5CLHSgRx3e6hMpU_z2o"
#define DATABASE_URL "https://trafficcontrol-7500e-default-rtdb.europe-west1.firebasedatabase.app/"
#define USER_EMAIL "kingoluwagbemiga@gmail.com"
#define USER_PASS "1234567890"

// Pin definitions for ESP32
#define SS_PIN 5 // SDA/SS pin
#define RST_PIN 22 // Reset pin

// Ultrasonic sensor pin definitions
#define TRIG_LANE1 12 // Lane 1 trigger pin
#define ECHO_LANE1 13 // Lane 1 echo pin
#define TRIG_LANE2 27 // Lane 2 trigger pin
#define ECHO_LANE2 14 // Lane 2 echo pin
#define TRIG_LANE3 25 // Lane 3 trigger pin
#define ECHO_LANE3 26 // Lane 3 echo pin
#define TRIG_LANE4 32 // Lane 4 trigger pin
#define ECHO_LANE4 33 // Lane 4 echo pin

// Create MFRC522 instance
MFRC522 mfrc522(SS_PIN, RST_PIN);

// User functions
void processData(AsyncResult &aResult);

// Authentication
UserAuth user_auth(Web_API_KEY, USER_EMAIL, USER_PASS);

// Firebase components
FirebaseApp app;
WiFiClientSecure ssl_client;
using AsyncClient = AsyncClientClass;
AsyncClient aClient(ssl_client);
RealtimeDatabase Database;

// Timer variables for sending data every 5 seconds
unsigned long lastSendTime = 0;
const unsigned long sendInterval = 5000; // 5 seconds in milliseconds

// Variable to save USER UID
String uid;

// Database main path (to be updated in setup with the user UID)
String databasePath;
// Database child nodes
String activeLanePath = "/activeLane";
String emergencyStatusPath = "/emergencyStatus";
String emergencyLanePath = "/emergencyLane";
String congestionStatusPath = "/congestionStatus";
String congestionLanePath = "/congestionLane";
String timePath = "/timestamp";

// Parent Node (to be updated in every loop)
String parentPath;

int timestamp;
const char* ntpServer = "pool.ntp.org";

// Timing variables
unsigned long laneActiveTime = 0;
const unsigned long laneDuration = 5000; // 5 seconds

// Ultrasonic sensor variables
const float triggerDistance = 5.1; // 5.1 cm trigger distance
const unsigned long triggerDuration = 3000; // 3 seconds
unsigned long ultrasonicTimers[4] = {0, 0, 0, 0}; // Timer for each lane
bool ultrasonicTriggered[4] = {false, false, false, false}; // Triggered state for each lane

// State variables
int currentLane = 1;
bool laneActive = false;
bool emergencyMode = false;
int emergencyLane = 0;
bool ultrasonicMode = false;
int ultrasonicLane = 0;

// Arduino communication variables
int lastSentLane = 0;
unsigned long lastArduinoSendTime = 0;
const unsigned long arduinoSendInterval = 100; // Send to Arduino every 100ms for responsiveness

// Create JSON objects for storing data
object_t jsonData, obj1, obj2, obj3, obj4, obj5, obj6;
JsonWriter writer;

// Initialize WiFi
void initWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi ..");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(1000);
  }
  Serial.println();
  Serial.print("Connected with IP: ");
  Serial.println(WiFi.localIP());
}

// Function that gets current epoch time
unsigned long getTime() {
  time_t now;
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    return(0);
  }
  time(&now);
  return now;
}

// Function to send lane information to Arduino UNO
void sendLaneToArduino(int lane) {
  if (lane != lastSentLane) {
    Serial2.print("LANE:");
    Serial2.println(lane);
    Serial.print("Sent to Arduino: LANE:");
    Serial.println(lane);
    lastSentLane = lane;
  }
}

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial); // Wait for serial port to connect
  
  // Initialize Serial2 for Arduino communication (GPIO 16=RX, GPIO 17=TX)
  Serial2.begin(9600, SERIAL_8N1, 16, 17);
  Serial.println("Serial communication with Arduino initialized");
  
  // Initialize WiFi and Firebase
  initWiFi();
  configTime(0, 0, ntpServer);

  // Configure SSL client
  ssl_client.setInsecure();
  ssl_client.setHandshakeTimeout(5);

  // Initialize Firebase
  initializeApp(aClient, app, getAuth(user_auth), processData, "🔐 authTask");
  app.getApp<RealtimeDatabase>(Database);
  Database.url(DATABASE_URL);
  
  // Initialize ultrasonic sensor pins
  pinMode(TRIG_LANE1, OUTPUT);
  pinMode(ECHO_LANE1, INPUT);
  pinMode(TRIG_LANE2, OUTPUT);
  pinMode(ECHO_LANE2, INPUT);
  pinMode(TRIG_LANE3, OUTPUT);
  pinMode(ECHO_LANE3, INPUT);
  pinMode(TRIG_LANE4, OUTPUT);
  pinMode(ECHO_LANE4, INPUT);
  
  // Initialize SPI bus
  SPI.begin();
  
  // Initialize MFRC522
  mfrc522.PCD_Init();
  
  // Optional: Show MFRC522 reader details
  mfrc522.PCD_DumpVersionToSerial();
  
  Serial.println("RC522 RFID and Ultrasonic Lane Controller with Firebase and Arduino Communication Initialized");
  Serial.println("===============================================================================================");
  Serial.println("Continuous sequence: Lane 1 -> 2 -> 3 -> 4 (no delays)");
  Serial.println("Scan RFID card for emergency override");
  Serial.println("Ultrasonic trigger: <5.1cm for 3+ seconds");
  Serial.println();
  
  // Start the sequence
  activateLane(1);
}

void loop() {
  unsigned long currentMillis = millis();
  
  // Maintain authentication and async tasks
  app.loop();
  
  // Check for RFID card (only if not in emergency mode)
  if (!emergencyMode) {
    checkForRFIDCard();
  }
  
  // Check ultrasonic sensors (only if not in emergency or ultrasonic mode)
  if (!emergencyMode && !ultrasonicMode) {
    checkUltrasonicSensors();
  }
  
  // Send lane information to Arduino periodically
  if (currentMillis - lastArduinoSendTime >= arduinoSendInterval) {
    int activeLane = getCurrentActiveLane();
    if (activeLane > 0) {
      sendLaneToArduino(activeLane);
    }
    lastArduinoSendTime = currentMillis;
  }
  
  // Handle lane timing
  if (laneActive && (currentMillis - laneActiveTime >= laneDuration)) {
    // Turn off current lane and handle next action
    if (emergencyMode) {
      Serial.print("EMERGENCY LANE ");
      Serial.print(emergencyLane);
      Serial.println(" - OFF");
      
      // Reset emergency mode
      emergencyMode = false;
      emergencyLane = 0;
      
      // Send "no active lane" momentarily to Arduino
      sendLaneToArduino(0);
      delay(100); // Brief pause
      
      // Resume normal traffic flow from the next lane in sequence
      currentLane++;
      if (currentLane > 4) {
        currentLane = 1;
      }
      
      Serial.print("Resuming normal traffic flow from Lane ");
      Serial.println(currentLane);
      
      // Activate the next lane in normal sequence
      activateLane(currentLane);
      
      // Send status update
      sendStatusToFirebase();
      
    } else if (ultrasonicMode) {
      Serial.print("ULTRASONIC LANE ");
      Serial.print(ultrasonicLane);
      Serial.println(" - OFF");
      
      // Reset ultrasonic mode
      ultrasonicMode = false;
      ultrasonicLane = 0;
      
      // Send "no active lane" momentarily to Arduino
      sendLaneToArduino(0);
      delay(100); // Brief pause
      
      // Resume normal traffic flow from the next lane in sequence
      currentLane++;
      if (currentLane > 4) {
        currentLane = 1;
      }
      
      Serial.print("Resuming normal traffic flow from Lane ");
      Serial.println(currentLane);
      
      // Activate the next lane in normal sequence
      activateLane(currentLane);
      
      // Send status update
      sendStatusToFirebase();
      
    } else {
      // Normal mode - turn off current lane
      Serial.print("Lane ");
      Serial.print(currentLane);
      Serial.println(" - OFF");
      
      // Immediately move to next lane in sequence
      currentLane++;
      if (currentLane > 4) {
        currentLane = 1;
      }
      
      // Activate next lane immediately (no delay)
      activateLane(currentLane);
    }
  }

  // Send periodic status updates to Firebase
  if (app.ready() && (currentMillis - lastSendTime >= sendInterval)) {
    sendStatusToFirebase();
    lastSendTime = currentMillis;
  }
}

// Function to get currently active lane
int getCurrentActiveLane() {
  if (laneActive) {
    if (emergencyMode) return emergencyLane;
    if (ultrasonicMode) return ultrasonicLane;
    return currentLane;
  }
  return 0; // No active lane
}

void sendStatusToFirebase() {
  if (!app.ready()) return;
  
  uid = app.getUid().c_str();
  
  // Update database path
  databasePath = "/TrafficControl/" + uid + "/status";
  
  //Get current timestamp
  timestamp = getTime();
  
  parentPath = databasePath + "/" + String(timestamp);
  
  // Determine active lane
  int activeLane = getCurrentActiveLane();
  
  // Create JSON object with status data
  writer.create(obj1, activeLanePath, activeLane);
  writer.create(obj2, emergencyStatusPath, emergencyMode);
  writer.create(obj3, emergencyLanePath, emergencyMode ? emergencyLane : 0);
  writer.create(obj4, congestionStatusPath, ultrasonicMode);
  writer.create(obj5, congestionLanePath, ultrasonicMode ? ultrasonicLane : 0);
  writer.create(obj6, timePath, timestamp);
  
  writer.join(jsonData, 6, obj1, obj2, obj3, obj4, obj5, obj6);
  
  Database.set<object_t>(aClient, parentPath, jsonData, processData, "RTDB_Send_Status");
  
  Serial.println("Status sent to Firebase:");
  Serial.print("Active Lane: "); Serial.println(activeLane);
  Serial.print("Emergency Mode: "); Serial.println(emergencyMode ? "YES" : "NO");
  if (emergencyMode) {
    Serial.print("Emergency Lane: "); Serial.println(emergencyLane);
  }
  Serial.print("Congestion Mode: "); Serial.println(ultrasonicMode ? "YES" : "NO");
  if (ultrasonicMode) {
    Serial.print("Congestion Lane: "); Serial.println(ultrasonicLane);
  }
  Serial.println("---");
}

void checkUltrasonicSensors() {
  unsigned long currentMillis = millis();
  
  // Check each ultrasonic sensor
  for (int lane = 1; lane <= 4; lane++) {
    float distance = measureDistance(lane);
    
    if (distance < triggerDistance && distance > 0) {
      // Distance is within trigger range
      if (!ultrasonicTriggered[lane - 1]) {
        // Start timing for this lane
        ultrasonicTimers[lane - 1] = currentMillis;
        ultrasonicTriggered[lane - 1] = true;
        Serial.print("Lane ");
        Serial.print(lane);
        Serial.print(" ultrasonic triggered - Distance: ");
        Serial.print(distance);
        Serial.println(" cm");
      } else {
        // Check if trigger duration has been met
        if (currentMillis - ultrasonicTimers[lane - 1] >= triggerDuration) {
          // Activate ultrasonic override
          ultrasonicMode = true;
          ultrasonicLane = lane;
          
          Serial.print("*** ULTRASONIC OVERRIDE ON LANE ");
          Serial.print(ultrasonicLane);
          Serial.print(" *** Distance: ");
          Serial.print(distance);
          Serial.println(" cm");
          
          activateLane(ultrasonicLane);
          
          // Reset the trigger for this lane
          ultrasonicTriggered[lane - 1] = false;
          ultrasonicTimers[lane - 1] = 0;
          
          // Send immediate status update to Firebase
          sendStatusToFirebase();
          
          break; // Exit the loop since we've activated a lane
        }
      }
    } else {
      // Distance is not within trigger range, reset timer
      if (ultrasonicTriggered[lane - 1]) {
        ultrasonicTriggered[lane - 1] = false;
        ultrasonicTimers[lane - 1] = 0;
        Serial.print("Lane ");
        Serial.print(lane);
        Serial.println(" ultrasonic trigger reset");
      }
    }
  }
}

float measureDistance(int lane) {
  int trigPin, echoPin;
  
  // Select the appropriate pins for the lane
  switch(lane) {
    case 1:
      trigPin = TRIG_LANE1;
      echoPin = ECHO_LANE1;
      break;
    case 2:
      trigPin = TRIG_LANE2;
      echoPin = ECHO_LANE2;
      break;
    case 3:
      trigPin = TRIG_LANE3;
      echoPin = ECHO_LANE3;
      break;
    case 4:
      trigPin = TRIG_LANE4;
      echoPin = ECHO_LANE4;
      break;
    default:
      return -1; // Invalid lane
  }
  
  // Send ultrasonic pulse
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // Read the echo pulse duration
  long duration = pulseIn(echoPin, HIGH, 30000); // 30ms timeout
  
  // Calculate distance in cm
  if (duration == 0) {
    return -1; // No echo received (timeout)
  }
  
  float distance = (duration * 0.034) / 2;
  return distance;
}

void checkForRFIDCard() {
  // Look for new cards
  if (!mfrc522.PICC_IsNewCardPresent()) {
    return;
  }
  
  // Select one of the cards
  if (!mfrc522.PICC_ReadCardSerial()) {
    return;
  }
  
  // Show card detected message
  Serial.println("Card Detected!");
  Serial.println("==============");
  
  // Display UID
  displayCardUID();
  
  // Check for emergency lane activation
  checkEmergencyLaneCard();
  
  Serial.println();
  
  // Halt PICC
  mfrc522.PICC_HaltA();
  
  // Stop encryption on PCD
  mfrc522.PCD_StopCrypto1();
}

void checkEmergencyLaneCard() {
  String scannedUID = getCardUID();
  
  int emergencyLaneRequested = 0;
  
  if (scannedUID == "CA85D800") {
    emergencyLaneRequested = 1;
  }
  else if (scannedUID == "1E4FE400") {
    emergencyLaneRequested = 2;
  }
  else if (scannedUID == "92F84D05") {
    emergencyLaneRequested = 3;
  }
  else if (scannedUID == "86C1E400") {
    emergencyLaneRequested = 4;
  }
  
  if (emergencyLaneRequested > 0) {
    // Emergency override activated
    emergencyMode = true;
    emergencyLane = emergencyLaneRequested;
    
    Serial.print("*** EMERGENCY ON LANE ");
    Serial.print(emergencyLane);
    Serial.println(" ***");
    
    activateLane(emergencyLane);
    
    // Send immediate status update to Firebase
    sendStatusToFirebase();
  } else {
    Serial.println("Unknown card - No emergency activation");
  }
}

void activateLane(int lane) {
  // Set the active lane
  if (!emergencyMode && !ultrasonicMode) {
    Serial.print("Lane ");
    Serial.print(lane);
    Serial.println(" - ON");
  }
  
  laneActive = true;
  laneActiveTime = millis();
  
  // Send lane change to Arduino immediately
  sendLaneToArduino(lane);
}

// Function to display card UID in multiple formats
void displayCardUID() {
  Serial.print("Card UID: ");
  
  // Display as HEX (most common format)
  String uidHex = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    if (mfrc522.uid.uidByte[i] < 0x10) {
      Serial.print("0");
      uidHex += "0";
    }
    Serial.print(mfrc522.uid.uidByte[i], HEX);
    uidHex += String(mfrc522.uid.uidByte[i], HEX);
    if (i < mfrc522.uid.size - 1) {
      Serial.print(":");
      uidHex += ":";
    }
  }
  Serial.println();
}

// Function to convert UID to string (useful for comparisons)
String getCardUID() {
  String uidString = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    if (mfrc522.uid.uidByte[i] < 0x10) {
      uidString += "0";
    }
    uidString += String(mfrc522.uid.uidByte[i], HEX);
  }
  uidString.toUpperCase();
  return uidString;
}

// Function to display card type
void displayCardType() {
  Serial.print("Card Type: ");
  
  MFRC522::PICC_Type piccType = mfrc522.PICC_GetType(mfrc522.uid.sak);
  Serial.println(mfrc522.PICC_GetTypeName(piccType));
  
  // Additional type information
  Serial.print("SAK: 0x");
  if (mfrc522.uid.sak < 0x10) Serial.print("0");
  Serial.println(mfrc522.uid.sak, HEX);
}

// Function to display additional card information
void displayCardInfo() {
  Serial.println("Additional Info:");
  
  // Check if it's a known card type
  MFRC522::PICC_Type piccType = mfrc522.PICC_GetType(mfrc522.uid.sak);
  
  switch (piccType) {
    case MFRC522::PICC_TYPE_MIFARE_MINI:
      Serial.println("- Mifare Mini, 320 bytes");
      break;
    case MFRC522::PICC_TYPE_MIFARE_1K:
      Serial.println("- Mifare 1KB");
      break;
    case MFRC522::PICC_TYPE_MIFARE_4K:
      Serial.println("- Mifare 4KB");
      break;
    case MFRC522::PICC_TYPE_MIFARE_UL:
      Serial.println("- Mifare Ultralight");
      break;
    case MFRC522::PICC_TYPE_ISO_14443_4:
      Serial.println("- ISO 14443-4 compliant");
      break;
    case MFRC522::PICC_TYPE_ISO_18092:
      Serial.println("- ISO 18092 compliant");
      break;
    default:
      Serial.println("- Unknown type");
      break;
  }
}

void processData(AsyncResult &aResult){
  if (!aResult.isResult())
    return;

  if (aResult.isEvent())
    Firebase.printf("Event task: %s, msg: %s, code: %d\n", aResult.uid().c_str(), aResult.eventLog().message().c_str(), aResult.eventLog().code());

  if (aResult.isDebug())
    Firebase.printf("Debug task: %s, msg: %s\n", aResult.uid().c_str(), aResult.debug().c_str());

  if (aResult.isError())
    Firebase.printf("Error task: %s, msg: %s, code: %d\n", aResult.uid().c_str(), aResult.error().message().c_str(), aResult.error().code());

  if (aResult.available())
    Firebase.printf("task: %s, payload: %s\n", aResult.uid().c_str(), aResult.c_str());
}
#define ENABLE_USER_AUTH
#define ENABLE_DATABASE

#include <SPI.h>
#include <MFRC522.h>
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <FirebaseClient.h>
#include "time.h"

// Network and Firebase credentials
#define WIFI_SSID "Galaxy A04s95E1"
#define WIFI_PASSWORD "babyboogal00"

#define Web_API_KEY "AIzaSyA-8_tGoa1Mm5HW5CLHSgRx3e6hMpU_z2o"
#define DATABASE_URL "https://trafficcontrol-7500e-default-rtdb.europe-west1.firebasedatabase.app/"
#define USER_EMAIL "kingoluwagbemiga@gmail.com"
#define USER_PASS "1234567890"

// Pin definitions for ESP32
#define SS_PIN 5 // SDA/SS pin
#define RST_PIN 22 // Reset pin

// Ultrasonic sensor pin definitions
#define TRIG_LANE1 12 // Lane 1 trigger pin
#define ECHO_LANE1 13 // Lane 1 echo pin
#define TRIG_LANE2 27 // Lane 2 trigger pin
#define ECHO_LANE2 14 // Lane 2 echo pin
#define TRIG_LANE3 25 // Lane 3 trigger pin
#define ECHO_LANE3 26 // Lane 3 echo pin
#define TRIG_LANE4 32 // Lane 4 trigger pin
#define ECHO_LANE4 33 // Lane 4 echo pin

// Create MFRC522 instance
//MFRC522 mfrc522(SS_PIN, RST_PIN);

// User functions
void processData(AsyncResult &aResult);

// Authentication
//UserAuth user_auth(Web_API_KEY, USER_EMAIL, USER_PASS);

// Firebase components
//FirebaseApp app;
//WiFiClientSecure ssl_client;
//using AsyncClient = AsyncClientClass;
//AsyncClient aClient(ssl_client);
//RealtimeDatabase Database;

// Timer variables for sending data every 5 seconds
//unsigned long lastSendTime = 0;
//const unsigned long sendInterval = 5000; // 5 seconds in milliseconds

// Variable to save USER UID
//String uid;

// Database main path (to be updated in setup with the user UID)
//String databasePath;
//// Database child nodes
//String activeLanePath = "/activeLane";
//String emergencyStatusPath = "/emergencyStatus";
//String emergencyLanePath = "/emergencyLane";
//String congestionStatusPath = "/congestionStatus";
//String congestionLanePath = "/congestionLane";
//String timePath = "/timestamp";
//
//// Parent Node (to be updated in every loop)
//String parentPath;
//
//int timestamp;
//const char* ntpServer = "pool.ntp.org";
//
//// Timing variables
//unsigned long laneActiveTime = 0;
//const unsigned long laneDuration = 5000; // 5 seconds
//
//// Ultrasonic sensor variables
//const float triggerDistance = 5.1; // 5.1 cm trigger distance
//const unsigned long triggerDuration = 3000; // 3 seconds
//unsigned long ultrasonicTimers[4] = {0, 0, 0, 0}; // Timer for each lane
//bool ultrasonicTriggered[4] = {false, false, false, false}; // Triggered state for each lane
//
//// State variables
//int currentLane = 1;
//bool laneActive = false;
//bool emergencyMode = false;
//int emergencyLane = 0;
//bool ultrasonicMode = false;
//int ultrasonicLane = 0;
//
//// Arduino communication variables
//int lastSentLane = 0;
//unsigned long lastArduinoSendTime = 0;
//const unsigned long arduinoSendInterval = 100; // Send to Arduino every 100ms for responsiveness
//
//// Create JSON objects for storing data
//object_t jsonData, obj1, obj2, obj3, obj4, obj5, obj6;
//JsonWriter writer;

// Initialize WiFi
8void initWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi ..");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(1000);
  }
  Serial.println();
  Serial.print("Connected with IP: ");
  Serial.println(WiFi.localIP());
}

// Function that gets current epoch time
unsigned long getTime() {
  time_t now;
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    return(0);
  }
  time(&now);
  return now;
}

// Function to send lane information to Arduino UNO
void sendLaneToArduino(int lane) {
  if (lane != lastSentLane) {
    Serial2.print("LANE:");
    Serial2.println(lane);
    Serial.print("Sent to Arduino: LANE:");
    Serial.println(lane);
    lastSentLane = lane;
  }
}

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial); // Wait for serial port to connect
  
  // Initialize Serial2 for Arduino communication (GPIO 16=RX, GPIO 17=TX)
  Serial2.begin(9600, SERIAL_8N1, 16, 17);
  Serial.println("Serial communication with Arduino initialized");
  
  // Initialize WiFi and Firebase
  initWiFi();
  configTime(0, 0, ntpServer);

  // Configure SSL client
  ssl_client.setInsecure();
  ssl_client.setHandshakeTimeout(5);

  // Initialize Firebase
  initializeApp(aClient, app, getAuth(user_auth), processData, "🔐 authTask");
  app.getApp<RealtimeDatabase>(Database);
  Database.url(DATABASE_URL);
  
  // Initialize ultrasonic sensor pins
  pinMode(TRIG_LANE1, OUTPUT);
  pinMode(ECHO_LANE1, INPUT);
  pinMode(TRIG_LANE2, OUTPUT);
  pinMode(ECHO_LANE2, INPUT);
  pinMode(TRIG_LANE3, OUTPUT);
  pinMode(ECHO_LANE3, INPUT);
  pinMode(TRIG_LANE4, OUTPUT);
  pinMode(ECHO_LANE4, INPUT);
  
  // Initialize SPI bus
  SPI.begin();
  
  // Initialize MFRC522
  mfrc522.PCD_Init();
  
  // Optional: Show MFRC522 reader details
  mfrc522.PCD_DumpVersionToSerial();
  
  Serial.println("RC522 RFID and Ultrasonic Lane Controller with Firebase and Arduino Communication Initialized");
  Serial.println("===============================================================================================");
  Serial.println("Continuous sequence: Lane 1 -> 2 -> 3 -> 4 (no delays)");
  Serial.println("Scan RFID card for emergency override");
  Serial.println("Ultrasonic trigger: <5.1cm for 3+ seconds");
  Serial.println();
  
  // Start the sequence
  activateLane(1);
}

void loop() {
  unsigned long currentMillis = millis();
  
  // Maintain authentication and async tasks
  app.loop();
  
  // Check for RFID card (only if not in emergency mode)
  if (!emergencyMode) {
    checkForRFIDCard();
  }
  
  // Check ultrasonic sensors (only if not in emergency or ultrasonic mode)
  if (!emergencyMode && !ultrasonicMode) {
    checkUltrasonicSensors();
  }
  
  // Send lane information to Arduino periodically
  if (currentMillis - lastArduinoSendTime >= arduinoSendInterval) {
    int activeLane = getCurrentActiveLane();
    if (activeLane > 0) {
      sendLaneToArduino(activeLane);
    }
    lastArduinoSendTime = currentMillis;
  }
  
  // Handle lane timing
  if (laneActive && (currentMillis - laneActiveTime >= laneDuration)) {
    // Turn off current lane and handle next action
    if (emergencyMode) {
      Serial.print("EMERGENCY LANE ");
      Serial.print(emergencyLane);
      Serial.println(" - OFF");
      
      // Reset emergency mode
      emergencyMode = false;
      emergencyLane = 0;
      
      // Send "no active lane" momentarily to Arduino
      sendLaneToArduino(0);
      delay(100); // Brief pause
      
      // Resume normal traffic flow from the next lane in sequence
      currentLane++;
      if (currentLane > 4) {
        currentLane = 1;
      }
      
      Serial.print("Resuming normal traffic flow from Lane ");
      Serial.println(currentLane);
      
      // Activate the next lane in normal sequence
      activateLane(currentLane);
      
      // Send status update
      sendStatusToFirebase();
      
    } else if (ultrasonicMode) {
      Serial.print("ULTRASONIC LANE ");
      Serial.print(ultrasonicLane);
      Serial.println(" - OFF");
      
      // Reset ultrasonic mode
      ultrasonicMode = false;
      ultrasonicLane = 0;
      
      // Send "no active lane" momentarily to Arduino
      sendLaneToArduino(0);
      delay(100); // Brief pause
      
      // Resume normal traffic flow from the next lane in sequence
      currentLane++;
      if (currentLane > 4) {
        currentLane = 1;
      }
      
      Serial.print("Resuming normal traffic flow from Lane ");
      Serial.println(currentLane);
      
      // Activate the next lane in normal sequence
      activateLane(currentLane);
      
      // Send status update
      sendStatusToFirebase();
      
    } else {
      // Normal mode - turn off current lane
      Serial.print("Lane ");
      Serial.print(currentLane);
      Serial.println(" - OFF");
      
      // Immediately move to next lane in sequence
      currentLane++;
      if (currentLane > 4) {
        currentLane = 1;
      }
      
      // Activate next lane immediately (no delay)
      activateLane(currentLane);
    }
  }

  // Send periodic status updates to Firebase
  if (app.ready() && (currentMillis - lastSendTime >= sendInterval)) {
    sendStatusToFirebase();
    lastSendTime = currentMillis;
  }
}

// Function to get currently active lane
int getCurrentActiveLane() {
  if (laneActive) {
    if (emergencyMode) return emergencyLane;
    if (ultrasonicMode) return ultrasonicLane;
    return currentLane;
  }
  return 0; // No active lane
}

void sendStatusToFirebase() {
  if (!app.ready()) return;
  
  uid = app.getUid().c_str();
  
  // Update database path
  databasePath = "/TrafficControl/" + uid + "/status";
  
  //Get current timestamp
  timestamp = getTime();
  
  parentPath = databasePath + "/" + String(timestamp);
  
  // Determine active lane
  int activeLane = getCurrentActiveLane();
  
  // Create JSON object with status data
  writer.create(obj1, activeLanePath, activeLane);
  writer.create(obj2, emergencyStatusPath, emergencyMode);
  writer.create(obj3, emergencyLanePath, emergencyMode ? emergencyLane : 0);
  writer.create(obj4, congestionStatusPath, ultrasonicMode);
  writer.create(obj5, congestionLanePath, ultrasonicMode ? ultrasonicLane : 0);
  writer.create(obj6, timePath, timestamp);
  
  writer.join(jsonData, 6, obj1, obj2, obj3, obj4, obj5, obj6);
  
  Database.set<object_t>(aClient, parentPath, jsonData, processData, "RTDB_Send_Status");
  
  Serial.println("Status sent to Firebase:");
  Serial.print("Active Lane: "); Serial.println(activeLane);
  Serial.print("Emergency Mode: "); Serial.println(emergencyMode ? "YES" : "NO");
  if (emergencyMode) {
    Serial.print("Emergency Lane: "); Serial.println(emergencyLane);
  }
  Serial.print("Congestion Mode: "); Serial.println(ultrasonicMode ? "YES" : "NO");
  if (ultrasonicMode) {
    Serial.print("Congestion Lane: "); Serial.println(ultrasonicLane);
  }
  Serial.println("---");
}

void checkUltrasonicSensors() {
  unsigned long currentMillis = millis();
  
  // Check each ultrasonic sensor
  for (int lane = 1; lane <= 4; lane++) {
    float distance = measureDistance(lane);
    
    if (distance < triggerDistance && distance > 0) {
      // Distance is within trigger range
      if (!ultrasonicTriggered[lane - 1]) {
        // Start timing for this lane
        ultrasonicTimers[lane - 1] = currentMillis;
        ultrasonicTriggered[lane - 1] = true;
        Serial.print("Lane ");
        Serial.print(lane);
        Serial.print(" ultrasonic triggered - Distance: ");
        Serial.print(distance);
        Serial.println(" cm");
      } else {
        // Check if trigger duration has been met
        if (currentMillis - ultrasonicTimers[lane - 1] >= triggerDuration) {
          // Activate ultrasonic override
          ultrasonicMode = true;
          ultrasonicLane = lane;
          
          Serial.print("*** ULTRASONIC OVERRIDE ON LANE ");
          Serial.print(ultrasonicLane);
          Serial.print(" *** Distance: ");
          Serial.print(distance);
          Serial.println(" cm");
          
          activateLane(ultrasonicLane);
          
          // Reset the trigger for this lane
          ultrasonicTriggered[lane - 1] = false;
          ultrasonicTimers[lane - 1] = 0;
          
          // Send immediate status update to Firebase
          sendStatusToFirebase();
          
          break; // Exit the loop since we've activated a lane
        }
      }
    } else {
      // Distance is not within trigger range, reset timer
      if (ultrasonicTriggered[lane - 1]) {
        ultrasonicTriggered[lane - 1] = false;
        ultrasonicTimers[lane - 1] = 0;
        Serial.print("Lane ");
        Serial.print(lane);
        Serial.println(" ultrasonic trigger reset");
      }
    }
  }
}

float measureDistance(int lane) {
  int trigPin, echoPin;
  
  // Select the appropriate pins for the lane
  switch(lane) {
    case 1:
      trigPin = TRIG_LANE1;
      echoPin = ECHO_LANE1;
      break;
    case 2:
      trigPin = TRIG_LANE2;
      echoPin = ECHO_LANE2;
      break;
    case 3:
      trigPin = TRIG_LANE3;
      echoPin = ECHO_LANE3;
      break;
    case 4:
      trigPin = TRIG_LANE4;
      echoPin = ECHO_LANE4;
      break;
    default:
      return -1; // Invalid lane
  }
  
  // Send ultrasonic pulse
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // Read the echo pulse duration
  long duration = pulseIn(echoPin, HIGH, 30000); // 30ms timeout
  
  // Calculate distance in cm
  if (duration == 0) {
    return -1; // No echo received (timeout)
  }
  
  float distance = (duration * 0.034) / 2;
  return distance;
}

void checkForRFIDCard() {
  // Look for new cards
  if (!mfrc522.PICC_IsNewCardPresent()) {
    return;
  }
  
  // Select one of the cards
  if (!mfrc522.PICC_ReadCardSerial()) {
    return;
  }
  
  // Show card detected message
  Serial.println("Card Detected!");
  Serial.println("==============");
  
  // Display UID
  displayCardUID();
  
  // Check for emergency lane activation
  checkEmergencyLaneCard();
  
  Serial.println();
  
  // Halt PICC
  mfrc522.PICC_HaltA();
  
  // Stop encryption on PCD
  mfrc522.PCD_StopCrypto1();
}

void checkEmergencyLaneCard() {
  String scannedUID = getCardUID();
  
  int emergencyLaneRequested = 0;
  
  if (scannedUID == "CA85D800") {
    emergencyLaneRequested = 1;
  }
  else if (scannedUID == "1E4FE400") {
    emergencyLaneRequested = 2;
  }
  else if (scannedUID == "92F84D05") {
    emergencyLaneRequested = 3;
  }
  else if (scannedUID == "86C1E400") {
    emergencyLaneRequested = 4;
  }
  
  if (emergencyLaneRequested > 0) {
    // Emergency override activated
    emergencyMode = true;
    emergencyLane = emergencyLaneRequested;
    
    Serial.print("*** EMERGENCY ON LANE ");
    Serial.print(emergencyLane);
    Serial.println(" ***");
    
    activateLane(emergencyLane);
    
    // Send immediate status update to Firebase
    sendStatusToFirebase();
  } else {
    Serial.println("Unknown card - No emergency activation");
  }
}

void activateLane(int lane) {
  // Set the active lane
  if (!emergencyMode && !ultrasonicMode) {
    Serial.print("Lane ");
    Serial.print(lane);
    Serial.println(" - ON");
  }
  
  laneActive = true;
  laneActiveTime = millis();
  
  // Send lane change to Arduino immediately
  sendLaneToArduino(lane);
}

// Function to display card UID in multiple formats
void displayCardUID() {
  Serial.print("Card UID: ");
  
  // Display as HEX (most common format)
  String uidHex = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    if (mfrc522.uid.uidByte[i] < 0x10) {
      Serial.print("0");
      uidHex += "0";
    }
    Serial.print(mfrc522.uid.uidByte[i], HEX);
    uidHex += String(mfrc522.uid.uidByte[i], HEX);
    if (i < mfrc522.uid.size - 1) {
      Serial.print(":");
      uidHex += ":";
    }
  }
  Serial.println();
}

// Function to convert UID to string (useful for comparisons)
String getCardUID() {
  String uidString = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    if (mfrc522.uid.uidByte[i] < 0x10) {
      uidString += "0";
    }
    uidString += String(mfrc522.uid.uidByte[i], HEX);
  }
  uidString.toUpperCase();
  return uidString;
}

// Function to display card type
void displayCardType() {
  Serial.print("Card Type: ");
  
  MFRC522::PICC_Type piccType = mfrc522.PICC_GetType(mfrc522.uid.sak);
  Serial.println(mfrc522.PICC_GetTypeName(piccType));
  
  // Additional type information
  Serial.print("SAK: 0x");
  if (mfrc522.uid.sak < 0x10) Serial.print("0");
  Serial.println(mfrc522.uid.sak, HEX);
}

// Function to display additional card information
void displayCardInfo() {
  Serial.println("Additional Info:");
  
  // Check if it's a known card type
  MFRC522::PICC_Type piccType = mfrc522.PICC_GetType(mfrc522.uid.sak);
  
  switch (piccType) {
    case MFRC522::PICC_TYPE_MIFARE_MINI:
      Serial.println("- Mifare Mini, 320 bytes");
      break;
    case MFRC522::PICC_TYPE_MIFARE_1K:
      Serial.println("- Mifare 1KB");
      break;
    case MFRC522::PICC_TYPE_MIFARE_4K:
      Serial.println("- Mifare 4KB");
      break;
    case MFRC522::PICC_TYPE_MIFARE_UL:
      Serial.println("- Mifare Ultralight");
      break;
    case MFRC522::PICC_TYPE_ISO_14443_4:
      Serial.println("- ISO 14443-4 compliant");
      break;
    case MFRC522::PICC_TYPE_ISO_18092:
      Serial.println("- ISO 18092 compliant");
      break;
    default:
      Serial.println("- Unknown type");
      break;
  }
}

void processData(AsyncResult &aResult){
  if (!aResult.isResult())
    return;

  if (aResult.isEvent())
    Firebase.printf("Event task: %s, msg: %s, code: %d\n", aResult.uid().c_str(), aResult.eventLog().message().c_str(), aResult.eventLog().code());

  if (aResult.isDebug())
    Firebase.printf("Debug task: %s, msg: %s\n", aResult.uid().c_str(), aResult.debug().c_str());

  if (aResult.isError())
    Firebase.printf("Error task: %s, msg: %s, code: %d\n", aResult.uid().c_str(), aResult.error().message().c_str(), aResult.error().code());

  if (aResult.available())
    Firebase.printf("task: %s, payload: %s\n", aResult.uid().c_str(), aResult.c_str());
}
