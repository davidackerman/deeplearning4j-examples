package org.deeplearning4j.examples.sample;

import java.awt.event.MouseEvent;  
import java.awt.event.MouseMotionListener; 

public class MouseMotionEventDemo implements MouseMotionListener {


public void mouseMoved(MouseEvent e) {
saySomething("Mouse moved", e);
}

public void mouseDragged(MouseEvent e) {
saySomething("Mouse dragged", e);
}

void saySomething(String eventDescription, MouseEvent e) {
System.out.println(eventDescription 
+ " (" + e.getX() + "," + e.getY() + ")"
+ " detected on "
+ e.getComponent().getClass().getName());
}
}
