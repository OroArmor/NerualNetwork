package oroarmor.neuralnetwork.util;

public class Dim3 {

	public int x, y, z = 1;

	public Dim3(int x) {
		this(x, 1, 1);
	}

	public Dim3(int x, int y) {
		this(x, y, 1);
	}

	public Dim3(int x, int y, int z) {
		this.x = x;
		this.y = y;
		this.z = z;
	}

	@Override
	public String toString() {
		return "Dim3 [x=" + x + ", y=" + y + ", z=" + z + "]";
	}
}
