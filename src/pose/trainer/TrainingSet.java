package pose.trainer;

import pose.tensor.Tensor;

public class TrainingSet
{
	private Tensor image;
	private Tensor targetHeatmaps;

	public TrainingSet(Tensor image, Tensor targetHeatmaps)
	{
		this.image = image;
		this.targetHeatmaps = targetHeatmaps;
	}

	public Tensor getImage()
	{
		return image;
	}

	public Tensor getTargetHeatmaps()
	{
		return targetHeatmaps;
	}
}
