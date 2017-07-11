void drawGridLines(){

	// grid
	int g = 15;//abs(int(mouseX/20))+1;
	int s = min(g,abs(int(mouseY/20))+1);
	println(g + " " + s);
	stroke(255,0,0, 100);
	//for (int x=0; x<width; x+=g) line(x,0,x,height);
	//for (int y=0; y<height; y+=g) line(0,y,width,y);
	int i = 0;
	for (int x=0; x<width; x+=s){
		for (int y=0; y<height; y+=s){
			i = (i+1)%3;
			if (i==0)
				stroke(255,0,0, 100);
			else if (i==1)
				stroke(0,0,255, 100);
			else if (i==2)
				stroke(0,255,0, 100);

			rect(x,y,g,g);
		}
	}
}