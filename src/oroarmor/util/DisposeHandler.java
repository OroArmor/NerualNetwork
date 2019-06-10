package oroarmor.util;

import processing.core.PApplet;

public class DisposeHandler {
	public DisposeHandler() {
	}
	
	public void register(PApplet p) {
		p.registerMethod("dispose", this);
		
	}

	public void dispose() {
		System.out.println("dispose");
	}
}