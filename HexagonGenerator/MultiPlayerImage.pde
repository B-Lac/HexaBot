boolean multiplayerMode = true;
float[][] multiplayerX, multiplayerY;
float[] multiplayerRot;
int playersPerRow = 10;
int numPlayers = 8*playersPerRow;

void randomizePlayers(){
	
	multiplayerRot = new float[numPlayers/playersPerRow];
	for (int i=0; i<numPlayers/playersPerRow; i++)
		multiplayerRot[i] = random(PI);

	shearX = gauss(0, .01);
	shearY = gauss(0, .01);
}