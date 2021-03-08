const int enablePin = 13; 
const int directionPin = 12;

const int light1Pin = 4;
const int light2Pin = 5;

const int ms1Pin = 9;
const int ms2Pin = 10;
const int ms3Pin = 11;

const int dogPin = 6;
const int catPin = 7;

const byte numChars = 32;
char receivedChars[numChars];

boolean newData = false;

boolean enable = false;
boolean clockwise = false;
boolean dog = false;
boolean cat = false;

int resolutionStep = 16;

void setup() {
  //// Establish correct pins
  pinMode(enablePin, OUTPUT);
  pinMode(directionPin, OUTPUT);
  pinMode(ms1Pin, OUTPUT);
  pinMode(ms2Pin, OUTPUT);
  pinMode(ms3Pin, OUTPUT);
  pinMode(dogPin, OUTPUT);
  pinMode(catPin, OUTPUT);
  pinMode(light1Pin, OUTPUT);
  pinMode(light2Pin, OUTPUT);
  
  digitalWrite(enablePin, HIGH);
  digitalWrite(directionPin, LOW);
  digitalWrite(ms1Pin, LOW);
  digitalWrite(ms2Pin, LOW);
  digitalWrite(ms3Pin, LOW);
  digitalWrite(dogPin, HIGH);
  digitalWrite(catPin, HIGH);
  pinMode(light1Pin, LOW);
  pinMode(light2Pin, LOW);

  Serial.begin(9600); // Default connection rate for BT module
}

 
void loop() {
  //// Continously parse bluetooth messages and determine course of action for motor and lights
  recvWithStartEndMarkers();

  if (newData == true) {
    String state(receivedChars);
    
    if (state.indexOf("/OFF/") != -1) {
      enable = false;
    } else if (state.indexOf("/ON/") != -1) {
      enable = true;
    }
  
    if (state.indexOf("/C/") != -1) {
      clockwise = true;
    } else if (state.indexOf("/CC/") != -1) {
      clockwise = false;
    }

    if (state.indexOf("/16/") != -1) {
      resolutionStep = 16; // sixteenth
    } else if (state.indexOf("/8/") != -1) {
      resolutionStep = 8; // eighth
    } else if (state.indexOf("/4/") != -1 ) {
      resolutionStep = 4; // quarter
    } else if (state.indexOf("/2/") != -1) {
      resolutionStep = 2; // half
    } else if (state.indexOf("/0/") != -1) {
      resolutionStep = 0; // full
    }

    if (state.indexOf("/C1/") != -1) {
      dog = true;
    } else if (state.indexOf("/C2/") != -1) {
      cat = true;
    } else {
      dog = false;
      cat = false;
    }

    if (state.indexOf("/CLEAN/") != -1) {
      clockwise = true;      
      enable = false;
    }
  }
  
  newData = false;

  if (enable) {
    digitalWrite(enablePin, HIGH);
  } else {
    digitalWrite(enablePin, LOW);
  }

  if (clockwise) {
    digitalWrite(directionPin, HIGH);
  } else {
    digitalWrite(directionPin, LOW);
  }

  //// Resolution for the stepper motor driver used for motor speed
  if (resolutionStep == 16) {
    digitalWrite(ms1Pin, HIGH);
    digitalWrite(ms2Pin, HIGH);
    digitalWrite(ms3Pin, HIGH);    
  } else if (resolutionStep == 8) {
    digitalWrite(ms1Pin, HIGH);
    digitalWrite(ms2Pin, HIGH);
    digitalWrite(ms3Pin, LOW);    
  } else if (resolutionStep == 4) {
    digitalWrite(ms1Pin, LOW);
    digitalWrite(ms2Pin, HIGH);
    digitalWrite(ms3Pin, LOW);    
  } else if (resolutionStep == 2) {
    digitalWrite(ms1Pin, HIGH);
    digitalWrite(ms2Pin, LOW);
    digitalWrite(ms3Pin, LOW);    
  } else if (resolutionStep == 0) {
    digitalWrite(ms1Pin, LOW);
    digitalWrite(ms2Pin, LOW);
    digitalWrite(ms3Pin, LOW);    
  } else {
    digitalWrite(ms1Pin, LOW);
    digitalWrite(ms2Pin, LOW);
    digitalWrite(ms3Pin, LOW);
  }

  //// Specific light for specific object
  if (dog) {
    digitalWrite(light1Pin, HIGH);
    digitalWrite(light2Pin, LOW);
    digitalWrite(dogPin, LOW);
    digitalWrite(catPin, HIGH);
  } else if (cat) {
    digitalWrite(light1Pin, LOW);
    digitalWrite(light2Pin, HIGH);
    digitalWrite(dogPin, HIGH);
    digitalWrite(catPin, LOW);
  } else {
    digitalWrite(light1Pin, LOW);
    digitalWrite(light2Pin, LOW);
    digitalWrite(dogPin, HIGH);
    digitalWrite(catPin, HIGH);
  }
}



void recvWithStartEndMarkers() {
	//// Determines if the bluetooth message recieved is in the correct format to be parsed for information
    static boolean recvInProgress = false;
    static byte ndx = 0;
    char startMarker = '<';
    char endMarker = '>';
    char rc;
 
    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();

        if (recvInProgress == true) {
            if (rc != endMarker) {
                receivedChars[ndx] = rc;
                ndx++;
                if (ndx >= numChars) {
                    ndx = numChars - 1;
                }
            }
            else {
                receivedChars[ndx] = '\0'; // terminate the string
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        }

        else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}


void showNewData() {
	//// Print data recieved for debug purposes
    if (newData == true) {
        Serial.print("This just in ... ");
        Serial.println(receivedChars);
        newData = false;
    }
}
