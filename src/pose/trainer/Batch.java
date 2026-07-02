package pose.trainer;

import pose.layer.Tensor;

public class Batch
{
	private Tensor images;
	private Tensor heatmaps;

	public Batch(Tensor images, Tensor heatmaps)
	{
		this.images = images;
		this.heatmaps = heatmaps;
	}

	public Tensor getImages()
	{
		return images;
	}

	public Tensor getHeatmaps()
	{
		return heatmaps;
	}
}
