boolean saveImages = false;
final long timestamp = System.currentTimeMillis();
final int numImages = 500;
int currImage = 0;


void saveImage(){
	if (!saveImages || currImage >= numImages){
		exit();
		return;
	}

	String title = "generated/"+timestamp+"/"+nf(currImage,5);
	save(title+".png");
	println(title);
	currImage++;

	StringBuilder out = new StringBuilder();
	//ArrayList<float[]> wallBoxes;
	//ArrayList<float[]> playerBoxes;
	out.append(wallBoxes.size());
	out.append("\n");
	out.append(playerBoxes.size());
	out.append("\n");
	for (float[] box: allBoxes){
		for (int i=0; i<4; i++){
			out.append(round(box[i]));
			if (i<3) out.append(",");
		}
		out.append("\n");
	}
	String s = out.toString();
	saveStrings(title+".csv", new String[]{s.substring(0,s.length()-1)});

	multiplayerMode = !multiplayerMode;
}
