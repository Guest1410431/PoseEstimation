package window;

import java.util.concurrent.atomic.AtomicInteger;

public class VideoController
{
	public final AtomicInteger latestSeekFrame = new AtomicInteger(-1);
	public int frame = 0;
	
	public volatile boolean updatingUI = false;
	public volatile boolean playing = false;
	
	public VideoController(){}
}
