float polarExtend = float(W)/float(H);
boolean trimBlack = true;

// converts sketch to polar rep
void cartToPolar(){
	PImage polar = new PImage(W, W);

	for (int i=0; i<W; i++){
		for (int j=0; j<W; j++){
			float r = i/2;
			if (trimBlack)
				r /= polarExtend;
			float t = TWO_PI*j/float(W) * polarExtend;
			int x = int(r * cos(t) + W/2);
			int y = int(r * sin(t) + H/2);
			polar.set(i,j,get(x,y));
		}
	}
	tint(255);
	image(polar, 0, 0);

}
