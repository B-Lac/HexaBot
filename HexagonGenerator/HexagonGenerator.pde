String imgName = "img/1454722389-0001.png";
PImage img;
float trans = 0;
float trans1 = trans;
float trans2 = 255;
float[] poly;
PFont fontSmall, fontLarge;
boolean polarMode = true;
boolean boundBoxMode = false;
int W = 768;
int H = 480;
boolean changedConfig = true;

void setup(){	
	//randomSeed(0);
	img = loadImage(imgName);
	size(W, W);
	fontSmall = createFont("bumpitup.ttf", 16);
	fontLarge = createFont("bumpitup.ttf", 33);
	textAlign(LEFT, TOP);
	setWalls();
	//randomizeConfig();
	//frameRate(5);
	if (saveImages){
		polarMode = true;
		boundBoxMode = false;
	}

}

void draw(){
	background(0);
	generateHexagon();
	if (polarMode) cartToPolar();
	populateBoundingBoxLists();
	if (boundBoxMode) drawBoundingBoxes();
	//drawGridLines();
	/*
	if (changedConfig){
		printBoxes();
		changedConfig = false;
	}
	*/
	if (saveImages){
		saveImage();
		randomizeConfig();
	}
	//drawPolarCircleOnCartesian();
}

void drawPolarCircleOnCartesian(){
	if (polarMode) return;
	stroke(0,255,0, 100);
	strokeWeight(2);
	noFill();
	ellipse(W/2,H/2,H,H);
	float r = H/2;
	for (float t: new float[]{0, 1.2*PI})
		line(W/2,H/2,W/2+r*cos(t),H/2+r*sin(t));
}

void printBoxes(){
	for (float[] box: allBoxes){
		for (int i=0; i<4; i++){
			print(round(box[i]));
			if (i<3) print(",");
		}
		println();
	}
	println();
}

void mousePressed(){
	if (mouseButton == LEFT){
		trans1 = trans1==trans ? 255 : trans;
		trans2 = trans2==trans ? 255 : trans;
	}
	else{
		//printWallScale();
		//printColor();
		print((mouseX) + ", " + (mouseY) + ", ");
	}
}

void printColor(){
	color c = img.get(mouseX, mouseY);
	println("color("+int(red(c)) + ", " + int(green(c)) + ", " + int(blue(c))+");");
	println();
}

void keyPressed(){

	if (key==CODED){
		if (keyCode==ESC) exit();
		if (keyCode==LEFT) playerRot += .2;
		if (keyCode==RIGHT) playerRot -= .2;
		println(playerRot+rot);
	}
	else{
		if (key==' ') polarMode = !polarMode;
		if (key=='b') boundBoxMode = !boundBoxMode;
		if (key=='r') randomizeConfig();
		if (key=='s') saveCurrentFrame();
	}

}

float x(float scale){
	return float(mouseX) / width * scale;
}

float y(float scale){
	return float(mouseY) / height * scale;
}

void printWallScale(){
	float r = sqrt(sq(mouseX-width/2)+sq(mouseY-height/2));
	float factor = scale / width;
	println(r * factor);
}

int saveIdx = 0;
void saveCurrentFrame(){
	PImage frame = get(0,0,W,polarMode ? W : H);
	frame.save("save/"+timestamp + "-"+saveIdx+".png");
	saveIdx++;
}