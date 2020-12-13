import processing.serial.*;// シリアルライブラリを取り入れる
Serial myPort; // myPort（任意名）というインスタンスを用意
String data; // 文字列クラスdata
String temp; // 文字列クラスtemp
PrintWriter output; //ファイルに書き込むためのオブジェクト
int gr = 1 ;
int i ;
int kk = 0;
int cl = 0;
int speed = 10 ;

void setup()
{
  println(Serial.list()); //すべての利用可能なシリアルポートのリストを得る. println()を使ってtext windowにその情報を書き込む.
  //myportに実体化。シリアルポートのリストの0番目に書かれたポートと115200bpsの速さで通信する。
  //Serial(parent, portName, baudRate)
  myPort = new Serial(this, Serial.list()[0], 115200);  // [2]
  myPort.clear();//バッファを空にし、そこに格納されているすべてのデータを削除する。

  size( 400, 300); //画面サイズ
  background(0,0,0);　//背景の色設定

  stroke( 255, 255, 255); //画面に引く線の色設定
  strokeWeight(0.5); //線の太さの設定
  line( 0, height/5, width, height/5);//上記設定の線を指定の位置に引く。line(x1, y1, x2, y2)
  line( 0, height * 2/5, width, height * 2/5);
  line( 0, height * 3/5, width, height * 3/5);
  line( 0, height * 4/5, width, height * 4/5);

  frameRate(1000); //smtime 0.005 nara 200  [1]

  File desk = new File(System.getProperty("user.home"), "Desktop");
  File newdir = new File(desk + "\\voltdata");
  //File newdir = new File("C:\\Users\\user\\Desktop\\voltdata"); //save dir
  //フォルダ作成位置の指定して実体化
  newdir.mkdir();  //対象の位置にフォルダを作る

  String zero ;  //文字列クラスの定義
  for (i = 1 ; i <= 500 ; i++){
    if( i < 10 ){
      zero = "00" ;
    } else if ( i < 100){
      zero = "0" ;
    } else {
      zero = "" ;
    }
    String filepath = newdir+ "\\data" + zero+ i+".csv" ; //ファイル名を作成(data***.csv)
    File file = new File (filepath); //測定記録用のファイルを作るクラスの実体化
    if (file.exists()){
    }
    else { //その名前のファイルがなければ作成する。
      output = createWriter(filepath);
      output.println("0");
      break ;
    }
  }
}

void draw() {
  if ( myPort.available() > 0) { //available()<=Returns the number of bytes available.
    data = myPort.readStringUntil('\n'); //\nを読み込むまでnullを返す。

    float x = gr;
    if ( data != null) {
      temp = data.substring(0,5); //dataの中身のうち、0から5番目までの文字列を抜き出す。
      float y = Float.parseFloat(temp); //数字文字列tempをfloat型に変換する。
      float tx = map(gr/speed , 0, width, 0, width);
      float ty = map(y , 0, 5, height, 0);
      fill( 255, 165, 0);
      stroke( 255, 165, 0);
      strokeWeight(1);
      ellipse( tx, ty, 3, 3);

      println(temp);
      output.println(temp);
      output.flush();
    }
  }
  else {
    float zx = map(gr/speed , 0, width, 0, width);
    float zy = map(0 , 0, 5, height, 0);
    fill( 255, 255, 255);
    stroke( 255, cl, cl);
    strokeWeight(1);
    ellipse( zx, zy, 2, 2);
  }
  gr++;

  if(gr/speed > width){
    background(0,0,0);
    stroke( 255, 255, 255);
    strokeWeight(0.5);
    line( 0, height/5, width, height/5);
    line( 0, height * 2/5, width, height * 2/5);
    line( 0, height * 3/5, width, height * 3/5);
    line( 0, height * 4/5, width, height * 4/5);
    gr = 0;
  }
}

void keyReleased(){
  if(kk == 0){
  output.close();
  stroke( 255, 255, 255);
  strokeWeight(0.5);
  line( gr/speed, 0, gr/speed, height);
  kk = 1 ;
  cl = 255;
}
  else{
  i++;
  File desk = new File(System.getProperty("user.home"), "Desktop");
  File newdir = new File(desk + "\\voltdata");
  String zero;
  if( i < 10 ){
    zero = "00" ;
  } else if ( i < 100){
    zero = "0" ;
  } else {
    zero = "" ;
  }
  String next = newdir+ "\\data" + zero+ i+".csv" ;
  output = createWriter(next);
  output.println("0");


  background(0,0,0);
  stroke( 255, 255, 255);
  strokeWeight(0.5);
  line( 0, height/5, width, height/5);
  line( 0, height * 2/5, width, height * 2/5);
  line( 0, height * 3/5, width, height * 3/5);
  line( 0, height * 4/5, width, height * 4/5);
  gr = 0;

  stroke( 255, 0, 0);
  strokeWeight(2);
  line( gr/speed, 0, gr/speed, height);
  kk = 0;
  cl = 0;
  }

}
