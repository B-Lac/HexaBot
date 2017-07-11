// represents a wall in the game
class Wall{
	int edgeIdx;
	float outScale, thick;
	float[] x, xSheared;
	float[] y, ySheared;

	Wall(int edgeIdx, float outScale, float thick){
		this.edgeIdx = edgeIdx;
		this.outScale = outScale;
		this.thick = thick;
		computeWall();
	}

	void computeWall(){
		x = new float[4];
		y = new float[4];
		float factor = CENTER_SCALE * scale / W;
		float extend1 = outScale / factor;
		float extend2 = (outScale+thick) / factor;

		float x1 = poly[edgeIdx*2], y1 = poly[edgeIdx*2+1];
		float x2 = poly[edgeIdx*2+2], y2 = poly[edgeIdx*2+3];
		x[0] = x1*extend1;
		y[0] =y1*extend1;
		x[1] = x1*extend2;
		y[1] =y1*extend2;
		x[2] = x2*extend2;
		y[2] = y2*extend2;
		x[3] = x2*extend1;
		y[3] =y2*extend1;

		xSheared = new float[4];
		ySheared = new float[4];
		for (int i=0; i<4; i++){
			float xs = x[i] + shearX*y[i];
			float ys = y[i] + shearY*x[i];
			xSheared[i] = xs + W/2;
			ySheared[i] = ys + H/2;
		}
	}

	void drawWall(){
		color c = (edgeIdx%2==1) ? c1 : c2;
		noStroke();
		fill(c, trans2);

		beginShape();
		for (int i=0; i<4; i++){
			vertex(x[i],y[i]);
		}
		endShape();
	}
}