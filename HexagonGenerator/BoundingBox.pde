ArrayList<float[]> wallBoxes;
ArrayList<float[]> playerBoxes;
ArrayList<float[]> allBoxes;
int black = color(0,0,0);
int white = color(255,255,255);



void drawBoundingBoxes(){
	noFill();
	stroke(0,255,0, 100);
	strokeWeight(2);
	
	if (polarMode){
		//populateBoundingBoxLists();
		for (float[] box : allBoxes){
			beginShape();
			for (int i=0; i<4; i++) vertex(box[(i==0|i==3)?0:2], box[(i==0|i==1)?1:3]);
			endShape(CLOSE);
		}
	}

	// draw walls
	else{
		drawCartesianBox(playerX, playerY);
		for (Wall wall : walls) drawCartesianBox(wall.xSheared, wall.ySheared);
	}
}

void drawCartesianBox(float[] x, float[] y){
	float[] xPtr = new float[4];
	float[] yPtr = new float[4];
	if (x.length==4){
		squeezeToGameBox(x, y, xPtr, yPtr);
	}
	else{
		for (int i=0; i<4; i++){
			xPtr[i] = ((i<2)? min(x): max(x)) + W/2;
			yPtr[i] = ((i<2)? min(y): max(y)) + H/2;
		}
	}
	//beginShape();
	//for (int i=0; i<4; i++) vertex(xPtr[i], yPtr[i]);
	//endShape(CLOSE);

	float x0 = min(xPtr);
	float x1 = max(xPtr);
	float y0 = min(yPtr);
	float y1 = max(yPtr);

	beginShape();
	for (int i=0; i<4; i++) vertex((i==1||i==2)?x0:x1, (i==0||i==1)?y0:y1);
	endShape(CLOSE);
}

void squeezeToGameBox(float[] x, float[] y, float[] xPtr, float[] yPtr){
	if (outOfBounds(x, y, 0, 0, W, H)) return;

	for (int i=0; i<4; i++){
		xPtr[i] = x[i];
		yPtr[i] = y[i];
	}

	for (int i=0; i<4; i++){
		int k = i;
		int j = (i+1)%4;
		boolean kOOB = outOfBounds(x[k],y[k],0,0,W,H);
		boolean jOOB = outOfBounds(x[j],y[j],0,0,W,H);
		if (kOOB != jOOB){
			if (jOOB){ k=j; j=i; }
			// k is OOB
			float t=1;
			if (x[k]<0){
				t = lerpT(x[j],x[k],0);
			}
			else if (x[k]>=W){
				t = lerpT(x[j],x[k],W);
			}
			else if (y[k]<0){
				t = lerpT(y[j],y[k],0);
			}
			else if (y[k]>=H){
				t = lerpT(y[j],y[k],H);
			}
			xPtr[k] = lerp(x[j], x[k], t);
			yPtr[k] = lerp(y[j], y[k], t);
		}
	}
}

float lerpT(float start, float stop, float middle){
	return (middle-start)/(stop-start);
}

void populateBoundingBoxLists(){
	wallBoxes = new ArrayList<float[]>();
	playerBoxes = new ArrayList<float[]>();
	allBoxes = new ArrayList<float[]>();
	
	if (multiplayerMode)
		for (int i=0; i<numPlayers; i++)
			addBoundingBoxPlayerPolar(multiplayerX[i], multiplayerY[i]);
	else
		addBoundingBoxPlayerPolar(playerX, playerY);
	for (Wall wall : walls){
		addBoundingBoxesWallPolar(wall);
	}
	for (float[] box : wallBoxes) allBoxes.add(box);
	for (float[] box : playerBoxes) allBoxes.add(box);
}

void addBoundingBoxPlayerPolar(float[] playerX, float[] playerY){
	float x0 = Float.POSITIVE_INFINITY, y0 = Float.POSITIVE_INFINITY,
	x1 = Float.NEGATIVE_INFINITY, y1 = Float.NEGATIVE_INFINITY;

	for (int i=0; i<3; i++){
		float x = playerX[i] + shearX * playerY[i];
		float y = playerY[i] + shearY * playerX[i];
		float r = sqrt(sq(x)+sq(y))*2;
		float t = atan2(y, x)*W/TWO_PI/polarExtend;
		if (trimBlack) r *= polarExtend;
		x0 = min(x0, r);
		x1 = max(x1, r);
		if (i==1) y0 = min(y0, t);
		if (i==2) y1 = max(y1, t);
	}

	if (y1<y0) y1 += W/polarExtend;
	int m=2;x0-=m;x1+=m;y0-=m;y1+=m;
	while (y1 < W){
		if (y0 > 0){
			playerBoxes.add(new float[]{x0,y0,x1,y1});
		}
		y0 += W/polarExtend;
		y1 += W/polarExtend;
	}
}


void addBoundingBoxesWallPolar(Wall wall){
	if (outOfBounds(wall.xSheared, wall.ySheared, 0, 0, W, H)) return;

	float[] xp = new float[4];
	float[] yp = new float[4];
	computePolarCoords(wall, xp, yp);

	// add wall boxes
	float extend = W/polarExtend;
	for (int offset = 0; offset<=extend*2; offset+=extend){

		// ignore polar boxes outside of image
		if (outOfBounds(xp, yp, 0, -offset, W, W-offset)){
			continue;
		}

		// squeeze box
		float x0 = min(xp), y0 = min(yp)+offset, x1 = max(xp), y1 = max(yp)+offset;
		while (y0+1 < y1 && isBlack(x0,y0) && isBlack(x1,y0)) y0++;
		while (y1-1 > y0 && isBlack(x0,y1) && isBlack(x1,y1)) y1--;
		while (x0+1 < x1 && isBlack(x0,y0) && isBlack(x0,y1)) x0++;
		while (x1-1 > x0 && isBlack(x1,y0) && isBlack(x1,y1)) x1--;
		
		if (y1-y0 < 30)
			continue;

		wallBoxes.add(new float[]{x0, y0, x1, y1});
	}
}

boolean isBlack(float r, float t){
	int pix = get(int(r),int(t));
	return pix == black || pix == white || outOfBounds(r,t,0,0,W,W);
}

boolean computePolarCoords(Wall wall, float[] xPtr, float[] yPtr){
	return computePolarCoords(wall.xSheared.clone(), wall.ySheared.clone(), xPtr, yPtr);
}

boolean computePolarCoords(float[] x, float[] y, float[] xPtr, float[] yPtr){

	// compute bounding box
	float x0 = polarWallX0(x, y);
	float x1 = polarWallX1(x, y);
	if (trimBlack){
		x0 *= polarExtend;
		x1 *= polarExtend;
	}
	float y0 = polarWallY0(x, y)*W/TWO_PI/polarExtend;
	float y1 = polarWallY1(x, y)*W/TWO_PI/polarExtend;

	for (int i=0; i<4; i++){
		xPtr[i] = (i==1||i==2)?x0:x1;
		yPtr[i] = (i==0||i==1)?y0:y1;
	}

	return true;	
}

float polarWallX0(float[] x, float[] y){
	float mx = (x[0]+x[3])/2 - W/2;
	float my = (y[0]+y[3])/2 - H/2;
	float r = sqrt(sq(mx)+sq(my))*2;
	return r;
}

float polarWallX1(float[] x, float[] y){
	float r = Float.NEGATIVE_INFINITY;
	for (int i=1; i<=2; i++){
		float ri = sqrt(sq(x[i]-W/2) + sq(y[i]-H/2))*2;
		r = max(r,ri);
	}
	return r;
}

float polarWallY0(float[] x, float[] y){
	return atan2(y[1] - H/2, x[1] - W/2);
}

float polarWallY1(float[] x, float[] y){
	float y0 = polarWallY0(x,y);
	float y1 = atan2(y[2] - H/2, x[2] - W/2);
	if (y0 < y1) y1 -= TWO_PI;
	return y1;
}

boolean outOfBounds(float x, float y, float x0, float y0, float x1, float y1){
	return x < x0 || y < y0 || x >= x1 || y >= y1;
}

boolean outOfBounds(float[] x, float[] y, float x0, float y0, float x1, float y1){
	boolean all = false;
	for (int i=0; i<4; i++){
		if (outOfBounds(x[i],y[i],x0,y0,x1,y1)==all){
			return all;
		}
	}
	return !all;
}