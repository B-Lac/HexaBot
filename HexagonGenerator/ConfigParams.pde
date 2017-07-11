// geometry params
float rot = 0.96;
float scale = 1;
int numEdges = 6;
float shearX = 0;
float shearY = 0;
float playerRot = 2.55;
float playerShadowOffsetX = 0;
float playerShadowOffsetY = 3;
float[] playerX, playerY;

// timer params
int bestSecs = 85;
int bestFrms = 47;
int timeSecs = 3;
int timeFrms = 42;
float complete = computeComplete();

// Permanent params
final float CENTER_SCALE = 40;
final float CENTER_WEIGHT = 5;
final float PLAYER_SCALE = 55;
final float PLAYER_SIZE = 7;
final int TIMER_POLY_OFFSET = 35;
final float[] TIMER_POLY1 = new float[]{534, 0, 557, 32, 656, 32, 667, 52, 768, 52, 768, 0};
final float[] TIMER_POLY2 = new float[]{534-TIMER_POLY_OFFSET, 0, 557-TIMER_POLY_OFFSET, 32, 656-TIMER_POLY_OFFSET, 32, 667-TIMER_POLY_OFFSET, 52, 768, 52, 768, 0};
final float[] BEST_POLY = new float[]{0, 0, 0, 32, 188, 32, 211, 0};

// walls
ArrayList<Wall> walls;
void setWalls(){
	/*
	walls = new ArrayList<Wall>();
	setHexagonPoints();
	walls.add(new Wall(0, 0.16, 0.05));
	walls.add(new Wall(0, 0.39, 0.05));
	walls.add(new Wall(2, 0.16, 0.05));
	walls.add(new Wall(2, 0.39, 0.05));
	walls.add(new Wall(4, 0.16, 0.05));
	walls.add(new Wall(4, 0.39, 0.05));
	*/
	
	walls = new ArrayList<Wall>();
	initLevelPatterns();
	setHexagonPoints();
	LEVEL1.generatePattern(0, 0.16);
	randomizeConfig();
}

// randomization
void randomizeConfig(){
	rot = random(TWO_PI);
	scale = gauss(1, .1);
	shearX = gauss(0, .1);
	shearY = gauss(0, .1);
	playerRot = 4.75 + random(2.5);
	PALETTE_LEVEL1.setColors(random(1));
	//PALETTE_LEVEL2.setColors(random(1));
	//PALETTE_LEVEL3.setColors(random(1));
	bestSecs = int(random(100));
	bestFrms = int(random(60));
	timeSecs = int(random(60));
	timeFrms = int(random(60));
	complete = computeComplete();
	if (multiplayerMode) randomizePlayers();
	setHexagonPoints();
	if (!multiplayerMode) LEVEL1.generatePattern();
	changedConfig = true;
}

float gauss(float mean, float var){
	return mean + randomGaussian() * var;
}

float computeComplete(){
	return (timeSecs + float(timeFrms)/60.) / 60.;
}