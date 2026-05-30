package pose.layer;

public class UpSampleLayer extends Layer
{
	private final int scale;
	
	public UpSampleLayer(int scale)
	{
		this.scale = scale;
	}

	@Override
	public Tensor forward(Tensor input)
	{
		// TODO: Nearest Neighbors
		return input;
	}
}
