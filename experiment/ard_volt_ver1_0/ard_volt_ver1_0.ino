#include <MsTimer2.h>
#define smtime 0.005 // サンプリング周期
//double time = 0 ;

void sample(){
  //time += smtime ;
  double data = analogRead(A0) * 4.9 / 1024 ;//dataにA0ピンからアナログデータ(電圧値)を格納する。analogRead関数の値は5[V]/1024 = 0.0049[V] = 4.9[mV] の単位で、1ずつ変わる。
  if (data != 0 ){
    //Serial.println(time,3);
    Serial.println(data,3);//dataの値を小数点以下３桁で送信する。
  }   
}

void setup(){
  analogReference(DEFAULT);//アナログ入力で使われる基準電圧を電源電圧(5V)に設定する。
  Serial.begin(115200);//シリアル通信のデータ転送を115200bpsで行う。
  
  MsTimer2::set(smtime * 1000,sample);//smtime(0.005s)*1000msごとにsample関数を呼び出す。
  MsTimer2::start();//タイマー割り込み開始
}

void loop(){
  //空ループ
}
