class Pattern{
	int[][] wallIdx;
	float spacing;
	float thick;

	Pattern(float spacing, float thick, int[][] wallIdx){
		this.wallIdx = wallIdx;
		this.spacing = spacing;
		this.thick = thick;
	}

	void addWalls(int initIdx, float initDist){
		for (int i=0; i<wallIdx.length; i++){
			for (int j=0; j<wallIdx[i].length; j++){
				int idx = (wallIdx[i][j] + initIdx) % numEdges;
				walls.add(new Wall(idx, initDist + spacing*i, thick));
			}	
		}
	}
}

class LevelPattern{

	final int level;
	final private ArrayList<Pattern> patterns = new ArrayList<Pattern>();

	LevelPattern(int level){
		this.level = level;
	}

	void addPattern(float spacing, float thick, int[][] wallIdx){
		patterns.add(new Pattern(spacing, thick, wallIdx));
	}

	void generatePattern(){
		//generatePattern(int(random(numEdges)), random(.15,.25));
		generatePattern(int(random(numEdges)), random(.1,.25));
	}

	void generatePattern(int initIdx, float initDist){
		if (multiplayerMode)
			return;
		patterns.get(int(random(patterns.size()))).addWalls(initIdx, initDist);
	}
}


final LevelPattern LEVEL1 = new LevelPattern(1);
final LevelPattern LEVEL2 = new LevelPattern(2);
final LevelPattern LEVEL3 = new LevelPattern(3);

boolean levelsInitialized = false;
LevelPattern getLevelPattern(int level){
	switch (level){
		case 1:
			return LEVEL1;
		default:
			return null;
	}
}

void initLevelPatterns(){
	if (levelsInitialized)
		return;
	levelsInitialized = true;
	LEVEL1.addPattern(0.23, 0.05,new int[][]{
		{1,2,3,4,5},
		{0,1,2,4,5}
	});
	LEVEL1.addPattern(.23, .05, new int[][]{
		{0,2,4},
		{0,2,4}
	});
	LEVEL1.addPattern(.23, .05, new int[][]{
		{0,3},
		{0,2,4},
		{0,2,4}
	});
}
