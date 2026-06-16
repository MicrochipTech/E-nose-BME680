/*
  BME680 LIVE classifier — coffee / alcohol / garlic
  ────────────────────────────────────────────────────
  Based on your original working sketch structure.
  Decision tree retrained with alcohol replacing mandarin.
  Turbulence removed (relabeled into adjacent scents).
  F1_macro = 0.903

  Board:   Adafruit Feather M4 Express (SAMD51)
  Library: Adafruit_BME680

  Commands:  b = force baseline to current gasEma now
             h = help
*/

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME680.h>
#include <math.h>
#include <Adafruit_NeoPixel.h>

static const uint32_t BAUD           = 115200;
static const uint32_t SAMPLE_PERIOD_MS = 350;
static const uint16_t HEATER_TEMP_C  = 320;
static const uint16_t HEATER_DUR_MS  = 150;
static const uint8_t  TEMP_OS        = BME680_OS_2X;
static const uint8_t  HUM_OS         = BME680_OS_2X;
static const uint8_t  PRES_OS        = BME680_OS_4X;
static const uint8_t  IIR_FILT       = BME680_FILTER_SIZE_3;
static const float    WINDOW_S       = 20.0f;
static const uint32_t PRED_PERIOD_MS = 2000;
static const float    EMA_ALPHA_FAST = 0.10f;
static const float    EMA_ALPHA_BASE = 0.01f;
static const int      MAX_SAMPLES    = (int)(WINDOW_S * 1000.0f / SAMPLE_PERIOD_MS) + 4;

Adafruit_BME680 bme;
uint32_t nextSampleMs=0, nextPredMs=0;
bool     havePrev=false;
uint32_t prevMs=0;
float    prevGasOhm=0;
bool     emaInit=false;
float    gasEma=NAN, baseline=NAN;

float buf_hum[64],buf_gasLog[64],buf_gasSlope[64];
float buf_gasEma[64],buf_dropPct[64],buf_gasOhms[64];
int   bufN=0;

static float meanOf(const float* a,int n){float s=0;for(int i=0;i<n;i++)s+=a[i];return n?s/n:0;}
static float stdOf(const float* a,int n,float mu){float s=0;for(int i=0;i<n;i++){float d=a[i]-mu;s+=d*d;}return n>1?sqrtf(s/(n-1)):0;}
static void copySort(const float* src,float* dst,int n){
  for(int i=0;i<n;i++)dst[i]=src[i];
  for(int i=1;i<n;i++){float k=dst[i];int j=i-1;while(j>=0&&dst[j]>k){dst[j+1]=dst[j];j--;}dst[j+1]=k;}
}
static float pct(const float* s,int n,float p){
  if(n<=0)return 0;float idx=p*(n-1);int i0=(int)idx,i1=i0+1<n?i0+1:n-1;
  return s[i0]*(1-(idx-i0))+s[i1]*(idx-i0);
}
static void push(float hum,float gl,float sl,float ema,float drop,float ohm){
  if(bufN<MAX_SAMPLES){
    buf_hum[bufN]=hum;buf_gasLog[bufN]=gl;buf_gasSlope[bufN]=sl;
    buf_gasEma[bufN]=ema;buf_dropPct[bufN]=drop;buf_gasOhms[bufN]=ohm;bufN++;
  } else {
    for(int i=1;i<MAX_SAMPLES;i++){
      buf_hum[i-1]=buf_hum[i];buf_gasLog[i-1]=buf_gasLog[i];buf_gasSlope[i-1]=buf_gasSlope[i];
      buf_gasEma[i-1]=buf_gasEma[i];buf_dropPct[i-1]=buf_dropPct[i];buf_gasOhms[i-1]=buf_gasOhms[i];
    }
    int e=MAX_SAMPLES-1;
    buf_hum[e]=hum;buf_gasLog[e]=gl;buf_gasSlope[e]=sl;
    buf_gasEma[e]=ema;buf_dropPct[e]=drop;buf_gasOhms[e]=ohm;
  }
}

enum Feat{F_HUM_MIN=0,F_HUM_P10,F_GASLOG_P50,F_GASSLOPE_P50,
          F_GASEMA_MIN,F_GASEMA_P10,F_GASEMA_P90,F_DROPPCT_P50,F_GASOHMS_STD,F_GASOHMS_MIN};
enum Cls {CLS_ALCOHOL=0,CLS_COFFEE,CLS_GARLIC};
const char* CLS_NAME[]={"alcohol","coffee","garlic"};

static int predict_class(const float f[10]){
  if (f[F_GASLOG_P50] <= 11.6274f) {
    if (f[F_GASLOG_P50] <= 11.0239f) {
      if (f[F_GASSLOPE_P50] <= 0.0049f) {
        if (f[F_GASSLOPE_P50] <= 0.0019f) {
          return CLS_ALCOHOL;  // alcohol
        } else {
          if (f[F_GASOHMS_MIN] <= 23492.0000f) {
            return CLS_ALCOHOL;  // alcohol
          } else {
            return CLS_GARLIC;  // garlic
          }
        }
      } else {
        if (f[F_GASEMA_MIN] <= 18456.4824f) {
          if (f[F_GASOHMS_MIN] <= 13772.5000f) {
            return CLS_ALCOHOL;  // alcohol
          } else {
            return CLS_ALCOHOL;  // alcohol
          }
        } else {
          return CLS_GARLIC;  // garlic
        }
      }
    } else {
      if (f[F_GASOHMS_STD] <= 42816.2539f) {
        if (f[F_GASOHMS_MIN] <= 23161.0000f) {
          if (f[F_HUM_MIN] <= 47.0750f) {
            return CLS_ALCOHOL;  // alcohol
          } else {
            return CLS_GARLIC;  // garlic
          }
        } else {
          if (f[F_GASLOG_P50] <= 11.2167f) {
            return CLS_GARLIC;  // garlic
          } else {
            return CLS_GARLIC;  // garlic
          }
        }
      } else {
        if (f[F_GASOHMS_STD] <= 46458.2148f) {
          return CLS_ALCOHOL;  // alcohol
        } else {
          return CLS_ALCOHOL;  // alcohol
        }
      }
    }
  } else {
    if (f[F_HUM_P10] <= 48.8200f) {
      if (f[F_GASEMA_P10] <= 39984.4238f) {
        if (f[F_GASOHMS_STD] <= 54245.4395f) {
          return CLS_ALCOHOL;  // alcohol
        } else {
          if (f[F_GASOHMS_MIN] <= 10855.0000f) {
            return CLS_COFFEE;  // coffee
          } else {
            return CLS_COFFEE;  // coffee
          }
        }
      } else {
        if (f[F_HUM_P10] <= 48.1980f) {
          return CLS_COFFEE;  // coffee
        } else {
          if (f[F_GASEMA_MIN] <= 105962.7305f) {
            return CLS_COFFEE;  // coffee
          } else {
            return CLS_GARLIC;  // garlic
          }
        }
      }
    } else {
      if (f[F_GASEMA_MIN] <= 97708.7070f) {
        if (f[F_HUM_MIN] <= 48.5200f) {
          return CLS_COFFEE;  // coffee
        } else {
          return CLS_COFFEE;  // coffee
        }
      } else {
        return CLS_GARLIC;  // garlic
      }
    }
  }
}

static bool computeFeatures(float f[10]){
  int n=bufN;
  if(n<(int)(WINDOW_S*1000.0f/SAMPLE_PERIOD_MS*0.8f)) return false;
  float tmp[64];
  copySort(buf_hum,tmp,n);     f[F_HUM_MIN]=tmp[0]; f[F_HUM_P10]=pct(tmp,n,0.10f);
  copySort(buf_gasLog,tmp,n);  f[F_GASLOG_P50]=pct(tmp,n,0.50f);
  copySort(buf_gasSlope,tmp,n);f[F_GASSLOPE_P50]=pct(tmp,n,0.50f);
  copySort(buf_gasEma,tmp,n);  f[F_GASEMA_MIN]=tmp[0];f[F_GASEMA_P10]=pct(tmp,n,0.10f);f[F_GASEMA_P90]=pct(tmp,n,0.90f);
  copySort(buf_dropPct,tmp,n); f[F_DROPPCT_P50]=pct(tmp,n,0.50f);
  float mu=meanOf(buf_gasOhms,n);f[F_GASOHMS_STD]=stdOf(buf_gasOhms,n,mu);
  copySort(buf_gasOhms,tmp,n); f[F_GASOHMS_MIN]=tmp[0];
  return true;
}

static void sampleOnce(){
  if(!bme.performReading()){Serial.println("# WARN: read fail");return;}
  uint32_t ms=millis();
  float hum=bme.humidity, gasOhm=(float)bme.gas_resistance;
  float gasLog=logf(gasOhm>1?gasOhm:1);
  float dlog=0;
  if(havePrev){float dt=(ms-prevMs)/1000.0f;if(dt>0)dlog=(gasLog-logf(prevGasOhm>1?prevGasOhm:1))/dt;}
  prevMs=ms;prevGasOhm=gasOhm;havePrev=true;
  if(!emaInit){gasEma=gasOhm;baseline=gasOhm;emaInit=true;}
  else{gasEma=EMA_ALPHA_FAST*gasOhm+(1-EMA_ALPHA_FAST)*gasEma;
       baseline=EMA_ALPHA_BASE*gasOhm+(1-EMA_ALPHA_BASE)*baseline;}
  float drop=(!isnan(baseline)&&baseline>1)?(gasEma-baseline)/baseline:0;
  push(hum,gasLog,dlog,gasEma,drop,gasOhm);
}

static uint8_t findAddr(){
  uint8_t a[2]={0x76,0x77};
  for(int i=0;i<2;i++){Wire.beginTransmission(a[i]);if(Wire.endTransmission()==0)return a[i];}
  for(uint8_t x=1;x<127;x++){Wire.beginTransmission(x);if(Wire.endTransmission()==0)return x;}
  return 0;
}

void setup(){
  Serial.begin(BAUD);while(!Serial)delay(10);
  Serial.println("\nBME680 — coffee / alcohol / garlic classifier");
  Serial.println("Wait ~20s for window to fill, then send 'b' to lock baseline.");
  Wire.begin();
  uint8_t addr=findAddr();
  if(!addr){Serial.println("ERROR: no I2C");while(1);}
  if(!bme.begin(addr)){Serial.println("ERROR: BME680 init");while(1);}
  bme.setTemperatureOversampling(TEMP_OS);bme.setHumidityOversampling(HUM_OS);
  bme.setPressureOversampling(PRES_OS);bme.setIIRFilterSize(IIR_FILT);
  bme.setGasHeater(HEATER_TEMP_C,HEATER_DUR_MS);
  Serial.print("# BME680 0x");Serial.println(addr,HEX);
  nextSampleMs=millis();nextPredMs=millis()+PRED_PERIOD_MS;
}

void loop(){
  while(Serial.available()){
    char ch=(char)Serial.read();
    if(ch=='b'&&!isnan(gasEma)){baseline=gasEma;Serial.print("# BASELINE=");Serial.println(baseline,0);}
    if(ch=='h'){Serial.println("# b=force baseline  h=help");}
  }
  uint32_t now=millis();
  if((int32_t)(now-nextSampleMs)>=0){nextSampleMs+=SAMPLE_PERIOD_MS;sampleOnce();}
  if((int32_t)(now-nextPredMs)>=0){
    nextPredMs+=PRED_PERIOD_MS;
    float f[10];
    if(!computeFeatures(f)){
      Serial.print("# filling ");Serial.print(bufN);Serial.print("/");Serial.println(MAX_SAMPLES);return;
    }
    int cls=predict_class(f);

// DEMO
  // Humidity gate — overrides tree for coffee vs garlic
  // Coffee: hum_p10 always < 49.5%   Garlic: hum_p10 always > 50.9%
  // 1.5% clean gap in all recordings — zero overlap
  if (f[F_HUM_P10] > 50.0f) {
      cls = CLS_GARLIC;
  } else if (cls == CLS_GARLIC && f[F_HUM_P10] < 50.0f) {
      cls = CLS_COFFEE;
  }
  // if     (cls == CLS_COFFEE)  setPixel(180,  80,   0);  // change these
  // else if(cls == CLS_ALCOHOL) setPixel(220,   0,   0);  // change these
  // else if(cls == CLS_GARLIC)  setPixel(  0, 180,   0);  // change these
    Serial.print("# CLASS=");Serial.print(CLS_NAME[cls]);
    Serial.print("  gasOhm=");Serial.print(prevGasOhm,0);
    Serial.print("  gasEma=");Serial.print(isnan(gasEma)?0:gasEma,0);
    Serial.print("  baseline=");Serial.print(isnan(baseline)?0:baseline,0);
    Serial.print("  dropPct=");
    if(!isnan(baseline)&&baseline>1&&!isnan(gasEma))Serial.print((gasEma-baseline)/baseline,4);
    else Serial.print(0.0f,4);
    Serial.println();
  }
}
