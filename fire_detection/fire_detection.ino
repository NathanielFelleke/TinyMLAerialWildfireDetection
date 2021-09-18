#include <IridiumSBD.h>
#include <TinyGPS++.h>

TinyGPSPlus gps;

#define IridiumSerial Serial1

#define SLEEP_PIN 4
#define DIAGNOSTICS false

IridiumSBD modem(IridiumSerial, SLEEP_PIN);


//Serial config for gps
UART gpsSerial(digitalPinToPinName(3), digitalPinToPinName(2), NC, NC);
static const uint32_t GPSBaud = 9600;

//Import Tensorflow and Fire Detect Model
#include <TensorFlowLite.h>

#include "model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


#include <SPI.h>
#include <Wire.h>
#include <memorysaver.h>
// Arducam library
#include <ArduCAM.h>
// JPEGDecoder library
#include <JPEGDecoder.h>

#define MAX_JPEG_BYTES 8192
// The pin connected to the Arducam Chip Select
#define CS 7

//specify input dimensions for the model
#define imageWidth 160
#define imageHeight 120

// Camera library instance
ArduCAM myCAM(OV2640, CS);
// Temporary buffer for holding JPEG data from camera
uint8_t jpeg_buffer[MAX_JPEG_BYTES] = {0};
// Length of the JPEG data currently in the buffer
uint32_t jpeg_length = 0;

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 120 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


void setup() {
  int err;
  delay(1000);

  digitalWrite(LED_PWR, LOW); //turn off LED to keep power consumption low

  //Start the serial with the satellite modem
  IridiumSerial.begin(19200);


  err = modem.sleep();
  if (err != ISBD_SUCCESS)
  {
    Serial.print("sleep failed: error ");
    Serial.println(err);
  }




  //set up logging through error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;


  //map model schema to usable model
  model = tflite::GetModel(model_data);

  //import all ops that are needed for the model to run
  static tflite::MicroMutableOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddReduceMax();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddFullyConnected();


  //Build interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);

  
  gpsSerial.begin(GPSBaud);

  

  //Setup Arducam

  Wire.begin();
  // Configure the CS pin
  pinMode(CS, OUTPUT);
  digitalWrite(CS, HIGH);
  // initialize SPI
  SPI.begin();
  // Reset the CPLD
  myCAM.write_reg(0x07, 0x80);
  delay(100);
  myCAM.write_reg(0x07, 0x00);

  delay(100);
  // Test whether we can communicate with Arducam via SPI
  myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
  uint8_t test;
  test = myCAM.read_reg(ARDUCHIP_TEST1);
  if (test != 0x55) {
    TF_LITE_REPORT_ERROR(error_reporter, "Can't communicate with Arducam");
    delay(1000);
  }


  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  // Specify the smallest possible resolution (160 x 120 is also the input size for the model)
  myCAM.OV2640_set_JPEG_size(OV2640_160x120);
  delay(100);



}

void loop() {
  int err;

  //Get JPEG image from Arducam

  // Make sure the buffer is emptied before each capture
  myCAM.clear_bit(ARDUCHIP_GPIO, GPIO_PWDN_MASK);
  delay(50);
  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  // Start capture
  myCAM.start_capture();
  // Wait for indication that it is done
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
  }
  delay(50);
  // Clear the capture done flag
  myCAM.clear_fifo_flag();

  jpeg_length = myCAM.read_fifo_length();

  if (jpeg_length > MAX_JPEG_BYTES) {
    TF_LITE_REPORT_ERROR(error_reporter, "Too many bytes in FIFO buffer (%d)",
                         MAX_JPEG_BYTES);
  }
  if (jpeg_length == 0) {
    TF_LITE_REPORT_ERROR(error_reporter, "No data in Arducam FIFO buffer");
  }


  myCAM.CS_LOW();
  myCAM.set_fifo_burst();
  for (int index = 0; index < jpeg_length; index++) {
    jpeg_buffer[index] = SPI.transfer(0x00);
  }
  delayMicroseconds(15);
  myCAM.CS_HIGH();
  JpegDec.decodeArray(jpeg_buffer, jpeg_length);

  //const int keep_x_mcus = imageWidth / JpegDec.MCUWidth;
  //const int keep_y_mcus = imageHeight / JpegDec.MCUHeight;


  // Pointer to the current pixel
  uint16_t* pImg;
  // Color of the current pixel
  uint16_t color;

  while (JpegDec.read()) {

    // Pointer to the current pixel
    pImg = JpegDec.pImage;

    //help us find where we are in the image
    int x_origin = JpegDec.MCUx * JpegDec.MCUWidth;
    int y_origin = JpegDec.MCUy * JpegDec.MCUHeight;


    // Loop through the MCU's rows and columns
    for (int mcu_row = 0; mcu_row < JpegDec.MCUHeight; mcu_row++) {
      // The y coordinate of this pixel in the output index
      int current_y =  y_origin + mcu_row;
      for (int mcu_col = 0; mcu_col < JpegDec.MCUWidth; mcu_col++) {
        // Read the color of the pixel as 16-bit integer

        color = *pImg++;

        // Convert the RGB565 values to normal RGB values
        int r   = ((color >> 11) & 0x1f) << 3;
        int g = ((color >> 5) & 0x3f) << 2;
        int b  = ((color >> 0) & 0x1f) << 3;


        // Convert to signed 8-bit integer by subtracting 128
        r -= 128;
        g -= 128;
        b -= 128;


        // The x coordinate of this pixel in the output image
        int current_x =   x_origin + mcu_col;
        // The index of this pixel in our flat output buffer
        int index = (current_y * 160) + current_x;


        //Load the pixel colors into the input
        input->data.int8[3 * index] = static_cast<int8_t>(r);
        input->data.int8[3 * index + 1] = static_cast<int8_t>(g);
        input->data.int8[3 * index + 2] = static_cast<int8_t>(b);
      }
    }
  }

  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }



  Serial.println(output->data.int8[0]);

 

  if (output->data.int8[0] >= 100) { //If a fire is detected

    //Turning on the satellite and sending that there is a fire
    err = modem.begin();
    if (err != ISBD_SUCCESS)
    {
      Serial.print("Begin failed: error ");
      Serial.println(err);
      if (err == ISBD_NO_MODEM_DETECTED)
        Serial.println("No modem detected: check wiring.");
    }


    String currentLat;
    String currentLong;

      //get the GPS location 
      if (gps.location.isValid())
      {
        currentLat = String(gps.location.lat(), 6);
        currentLong = String(gps.location.lng(), 6);
      }
      else{

        currentLat = "Invalid";
        currentLong = "Invalid";
      }

    String message = "Fire Detected at" + currentLat + ", " + currentLong;
    err = modem.sendSBDText(message.c_str());
    if (err != ISBD_SUCCESS)
    {
      Serial.print("sendSBDText failed: error ");
      Serial.println(err);
      if (err == ISBD_SENDRECEIVE_TIMEOUT)
        Serial.println("Try again with a better view of the sky.");
    }

    //Putting the satellite back to sleep

    err = modem.sleep();
    if (err != ISBD_SUCCESS)
    {
      Serial.print("sleep failed: error ");
      Serial.println(err);
    }
    myCAM.set_bit(ARDUCHIP_GPIO, GPIO_PWDN_MASK); //Arducam Low Power Mode
    delay(500);


  }
//reading data from GPS
  
  while (gpsSerial.available() > 0)
    gps.encode(gpsSerial.read());
      

  if (millis() > 5000 && gps.charsProcessed() < 10)
  {
    Serial.println(F("No GPS detected: check wiring."));
    while(true);
  }
}
