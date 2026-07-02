package pose.trainer;

public class DataInfo
{
	private final String imageTensorPath;
	private final String heatmapTensorPath;

	public DataInfo(String imageTensorPath, String heatmapTensorPath)
	{
		this.imageTensorPath = imageTensorPath;
		this.heatmapTensorPath = heatmapTensorPath;
	}

	public String getImageTensorPath()
	{
		return imageTensorPath;
	}

	public String getHeatmapTensorPath()
	{
		return heatmapTensorPath;
	}
}
