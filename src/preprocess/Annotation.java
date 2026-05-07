package preprocess;

public class Annotation
{
	private int jointId;
	private int xPos;
	private int yPos;
	private int visibility;

	public Annotation(int jointId, int xPos, int yPos, int visibility)
	{
		this.jointId = jointId;
		this.xPos = xPos;
		this.yPos = yPos;
		this.visibility = visibility;
	}

	public int getJointId()
	{
		return jointId;
	}

	public void setJointId(int jointId)
	{
		this.jointId = jointId;
	}

	public int getxPos()
	{
		return xPos;
	}

	public void setxPos(int xPos)
	{
		this.xPos = xPos;
	}

	public int getyPos()
	{
		return yPos;
	}

	public void setyPos(int yPos)
	{
		this.yPos = yPos;
	}

	public int getVisibility()
	{
		return visibility;
	}

	public void setVisibility(int visibility)
	{
		this.visibility = visibility;
	}
}
