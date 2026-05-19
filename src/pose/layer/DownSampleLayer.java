package pose.layer;

public class DownSampleLayer extends Layer
{
	private final int poolSize;
	private final int stride;
	
	public DownSampleLayer(int poolSize, int stride)
	{
		super();
		this.poolSize = poolSize;
		this.stride = stride;
	}

	@Override
	public Tensor forward(Tensor input)
	{
		// TODO: Pooling
		return input;
	}
}
