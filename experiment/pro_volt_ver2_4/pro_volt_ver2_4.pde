import processing.serial.*;
Serial myPort; 
String data; 
String temp;
PrintWriter output; 
int gr = 1 ;
int i ;
int kk = 0;
int cl = 0;
int speed = 10 ;

void setup()
{
  println(Serial.list());
  myPort = new Serial(this, Serial.list()[0], 115200);  // [2]
  myPort.clear();
  
  size( 400, 300);
  background(0,0,0);
  
  stroke( 255, 255, 255);
  strokeWeight(0.5);
  line( 0, height/5, width, height/5);
  line( 0, height * 2/5, width, height * 2/5);
  line( 0, height * 3/5, width, height * 3/5);
  line( 0, height * 4/5, width, height * 4/5);

  frameRate(1000); //smtime 0.005 nara 200  [1]
  
  File desk = new File(System.getProperty("user.home"), "Desktop"); 
  File newdir = new File(desk + "\\voltdata"); 
  //File newdir = new File("C:\\Users\\user\\Desktop\\voltdata"); //save dir 
  newdir.mkdir();
  
  String zero ;  
  for (i = 1 ; i <= 500 ; i++){
    if( i < 10 ){
      zero = "00" ;
    } else if ( i < 100){
      zero = "0" ;
    } else {
      zero = "" ;
    }
    String filepath = newdir+ "\\data" + zero+ i+".csv" ; 
    File file = new File (filepath); 
    if (file.exists()){
    }
    else {
      output = createWriter(filepath);
      output.println("0");
      break ;
    }
  }
}

void draw() {
  if ( myPort.available() > 0) {
    data = myPort.readStringUntil('\n');
      
    float x = gr;
    if ( data != null) {    
      temp = data.substring(0,5);
      float y = Float.parseFloat(temp);
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
  
  