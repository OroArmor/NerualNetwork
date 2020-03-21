import processing.core.PApplet;
import processing.core.PImage;

public class HalfMill extends PApplet {
	public static void main(String[] args) {
		PApplet.main("HalfMill");
	}

	public void settings() {
		size(1000 + (int) random(-20, 20), 500 + (int) random(-10, 10));
	}

	public void setup() {
		System.out.println(width + " " + height);
		noStroke();
		int counter = 0;
		int res = 1;
		for (int i = 0; i < width; i += res) {
			for (int j = 0; j < height; j += res) {
				int black = (counter++) % 2;
				int bwColor = color((black * 255));
				fill(bwColor);
				rect(i, j, res, res);
			}
			if (height % 2 == 0) {
				counter++;
			}
		}
		PImage screen = g.copy();
		screen.save("C:\\Users\\s-oronae\\Desktop\\Test.png");
	}
}
