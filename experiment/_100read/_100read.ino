#include <stdio.h>

int main(){
  FILE *fp;
  char *fname = "100ave.csv";
  int ret;
  char buf[1][10];
  double data[1];

  fp = fopen(fname,"r");
  if(fp == NULL){
    printf("%sファイルが開けません\n", fname);
    return -1;
  }

  printf("\n");

  fscanf(fp,"%s",buf[0]);
  printf("%s\n",buf[0]);


  while((ret=fscanf(fp, "%lf",&data[0]))!= EOF){
    printf("%lf\n",data[0]);
  }

  printf("\n");
  fclose(fp);
}

