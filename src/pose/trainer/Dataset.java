package pose.trainer;

import java.util.Collections;
import java.util.List;

public class Dataset
{
	private List<DataInfo> dataInfo;

	public Dataset(List<DataInfo> dataInfo)
	{
		this.dataInfo = dataInfo;
	}

	public int size()
	{
		return dataInfo.size();
	}

	public DataInfo get(int index)
	{
		return dataInfo.get(index);
	}

	public void shuffle()
	{
		Collections.shuffle(dataInfo);
	}
}
