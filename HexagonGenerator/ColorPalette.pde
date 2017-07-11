color c1 = color(203, 95, 25);// wall color 1 (same as center)
color c2 = color(178, 83, 21);// wall color 2 (different than center)
color c3 = color(67, 32, 5);// background color (same as center)
color c4 = color(100, 48, 7);// background color (different than center) and incomplete bar
color c5 = color(224, 116, 39);// player color and complete bar

// a color palette for a level
class Palette{
	final int NUM_COLORS = 5;
	color[] cset1, cset2;
	boolean useHSB;
	Palette(color[] cset1, color[] cset2, boolean hsb){
		this.useHSB = hsb;
		this.cset1 = cset1;
		this.cset2 = cset2;
	}
	Palette(color[] cset1, color[] cset2){
		this(cset1, cset2, false);
	}
	void setColors(float t){
		if (useHSB){
			colorMode(HSB, 256);
		}
		c1 = lerpColor(cset1[0], cset2[0], t);
		c2 = lerpColor(cset1[1], cset2[1], t);
		c3 = lerpColor(cset1[2], cset2[2], t);
		c4 = lerpColor(cset1[3], cset2[3], t);
		c5 = lerpColor(cset1[4], cset2[4], t);
	}
}

// Level 1 palette
color l1c11 = color(255, 255, 32);
color l1c21 = color(189, 166, 15);
color l1c31 = color(62, 50, 5);
color l1c41 = color(92, 75, 8);
color l1c51 = color(224, 221, 28);

color l1c12 = color(193, 21, 30);
color l1c22 = color(169, 18, 26);
color l1c32 = color(71, 17, 5);
color l1c42 = color(106, 26, 7);
color l1c52 = color(224, 32, 48);

Palette PALETTE_LEVEL1 = new Palette(
	new color[]{l1c11, l1c21, l1c31, l1c41, l1c51},
	new color[]{l1c12, l1c22, l1c32, l1c42, l1c52});

color l2c11 = color(4, 253, 125);
color l2c21 = l2c11;
color l2c31 = color(0, 0, 0);
color l2c41 = color(1, 39, 19);
color l2c51 = l2c11;

color l2c12 = color(247, 131, 4);
color l2c22 = l2c12;
color l2c32 = color(0, 0, 0);
color l2c42 = color(39, 20, 0);
color l2c52 = l2c12;

Palette PALETTE_LEVEL2 = new Palette(
	new color[]{l2c11, l2c21, l2c31, l2c41, l2c51},
	new color[]{l2c12, l2c22, l2c32, l2c42, l2c52});

//240,251
color l3c11 = color(249, 70, 16);
color l3c21 = l3c11;
color l3c31 = color(82, 11, 0);
color l3c41 = color(99, 17, 0);
color l3c51 = l3c11;

color l3c12 = color(249, 16, 215);
color l3c22 = l3c12;
color l3c32 = color(82, 0, 78);
color l3c42 = color(99, 0, 89);
color l3c52 = l3c12;

Palette PALETTE_LEVEL3 = new Palette(
	new color[]{l3c11, l3c21, l3c31, l3c41, l3c51},
	new color[]{l3c12, l3c22, l3c32, l3c42, l3c52}, true);

