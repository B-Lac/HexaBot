void generateHexagon(){
	tint(255,255,255,trans1);
	image(img,0,0);
	
	pushMatrix();
	translate(W/2,H/2);
	shearX(shearX);
	shearY(shearY);

	drawBackground();
	drawRadialPolygon();
	drawWalls();
	if (multiplayerMode)
		drawMultiplayer();
	else
		drawPlayer();

	popMatrix();
	fill(0);
	rect(0,H,W,W-H);
	
	drawHUD();
}

// sets the points that define the inner hexagon
void setHexagonPoints(){
	walls = new ArrayList<Wall>();
	poly = new float[numEdges*2 + 2];
	float polyThick = CENTER_WEIGHT*scale/W;
	float polyScale = CENTER_SCALE*scale/W - polyThick/2;
	for (int i=0; i<numEdges; i++){
		float t = i / float(numEdges) * TWO_PI;
		float x = sin(t+rot) * scale * CENTER_SCALE;
		float y = cos(t+rot) * scale * CENTER_SCALE;
		poly[i*2] = x;
		poly[i*2+1] = y;
	}
	poly[numEdges*2] = poly[0];
	poly[numEdges*2+1] = poly[1];
	
	for (int i=0; i<numEdges; i++)
		walls.add(new Wall(i, polyScale, polyThick));
}

// draws inner hexagon
void drawRadialPolygon(){
	fill(c3, trans2);
	noStroke();
	beginShape();
	for (int i=0; i<numEdges; i++){
		vertex(poly[i*2],poly[i*2+1]);
	}
	endShape(CLOSE);
}

// draws background without walls or center hexagon
void drawBackground(){
	float extend = W/CENTER_SCALE*2;
	color currC = c3;
	strokeWeight(2);
	for (int i=0; i<numEdges; i++){
		float x1 = poly[i*2], y1 = poly[i*2+1];
		float x2 = poly[i*2+2], y2 = poly[i*2+3];
		fill(currC, trans2);
		stroke(currC, trans2);
		beginShape();
			vertex(x1,y1);
			vertex(x1*extend,y1*extend);
			vertex(x2*extend,y2*extend);
			vertex(x2,y2);
		endShape();
		currC = currC==c3 ? c4 : c3;
	}
}

// draws all walls in level
void drawWalls(){
	for (Wall wall: walls){
		wall.drawWall();
	}
}

void drawMultiplayer(){
	multiplayerX = new float[numPlayers][3];
	multiplayerY = new float[numPlayers][3];

	for (int p=0; p<numPlayers/playersPerRow; p++){
		float pRot = multiplayerRot[p];
		for (int iters=0; iters<playersPerRow; iters++){
			int pIdx = p*playersPerRow+iters;
			float pScale = 1 + p*.45/scale;

			float centerX = sin(pRot) * scale * PLAYER_SCALE * pScale;
			float centerY = cos(pRot) * scale * PLAYER_SCALE * pScale;
			noStroke();

			for (int i=0; i<3; i++){
				float t = i / 3. * TWO_PI;
				float squish = p==0? 1: 1-pScale/PI;
				float x = sin(t+pRot) * scale * PLAYER_SIZE * (i==0?squish:pScale);
				float y = cos(t+pRot) * scale * PLAYER_SIZE * (i==0?squish:pScale);
				multiplayerX[pIdx][i] = centerX + x;
				multiplayerY[pIdx][i] = centerY + y;
			}
			
			fill(c3,trans2);
			beginShape();
			for (int i=0; i<3; i++){
				vertex(multiplayerX[pIdx][i] + playerShadowOffsetX, multiplayerY[pIdx][i] + playerShadowOffsetY);
			}
			endShape();

			fill(c5,trans2);
			beginShape();
			for (int i=0; i<3; i++){
				vertex(multiplayerX[pIdx][i], multiplayerY[pIdx][i]);
			}
			endShape();
			pRot += TWO_PI/playersPerRow;
		}
	}
}

// draw player triangle
void drawPlayer(){
	float centerX = sin(playerRot) * scale * PLAYER_SCALE;
	float centerY = cos(playerRot) * scale * PLAYER_SCALE;
	noStroke();

	playerX = new float[3];
	playerY = new float[3];

	for (int i=0; i<3; i++){
		float t = i / 3. * TWO_PI;
		float x = sin(t+playerRot) * scale * PLAYER_SIZE;
		float y = cos(t+playerRot) * scale * PLAYER_SIZE;
		playerX[i] = centerX + x;
		playerY[i] = centerY + y;

	}
	
	fill(c3,trans2);
	beginShape();
	for (int i=0; i<3; i++){
		vertex(playerX[i] + playerShadowOffsetX, playerY[i] + playerShadowOffsetY);
	}
	endShape();

	fill(c5,trans2);
	beginShape();
	for (int i=0; i<3; i++){
		vertex(playerX[i], playerY[i]);
	}
	endShape();	
}



// draws time progress bar
void drawHUD(){
	fill(0, trans2);
	noStroke();
	int tpo = timeSecs<10 ? 0 : TIMER_POLY_OFFSET;
	drawPolygon(timeSecs<10 ? TIMER_POLY1 : TIMER_POLY2);
	drawPolygon(BEST_POLY);

	fill(255, trans2);
	textFont(fontSmall);
	text("BEST: "+nf(bestSecs,2)+":"+nf(bestFrms,2) , 15, 4);
	text("TIME", 567-tpo, 4);
	fill(215, trans2);
	text(":"+nf(timeFrms,2), 713, 22);
	
	textFont(fontLarge);
	text(timeSecs, 676-tpo, 1);

	drawProgressBar();
}

// draws arbitrary polygon of form {x1,y1,x2,y2,...,xn,yn}
void drawPolygon(float[] poly){
	beginShape();
	for (int i=0; i<poly.length/2; i++){
		vertex(poly[i*2], poly[i*2+1]);
	}
	endShape();
}

// draws progress bar on top left of hud
void drawProgressBar(){
	float[] b = new float[]{17, 26, 175, 28};
	float x = int(lerp(b[0],b[2],complete));
	strokeWeight(1);
	fill(c5, trans2);
	beginShape();
	vertex(b[0],b[1]);
	vertex(b[0],b[3]);
	vertex(x,b[3]);
	vertex(x,b[1]);
	endShape();

	fill(c4, trans2);
	beginShape();
	vertex(x,b[1]);
	vertex(x,b[3]);
	vertex(b[2],b[3]);
	vertex(b[2],b[1]);
	endShape();
}