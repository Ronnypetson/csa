from torch.utils.data import Dataset
from skimage import io, transform

class Dataset0(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        ''' len(data) '''
        return None

    def __getitem__(self,idx):
        ''' sample = data[idx]
            if self.transform: sample = self.transform(sample)
        '''
        return None

class Rescale(object):
    def __init__(self,output_size):
        assert isinstanceof(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,sample):
        image, landmarks = sample['image'], sample['landmarks']
        h,w = image.shape[:2]
        if isinstanceof(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image,(new_h,new_w))
        landmarks = landmarks*[new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}

class ToTensor(object):
    def __call__(self,sample):
        image = image.transpose((2,0,1))
        return {'image':torch.from_numpy(image),
                'landmarks':torch.from_numpy(landmarks)}

if __name__=='__main__':
    scale = Rescale(256)
    # transforms.Compose([Rescale(256)])
    # dataloader = DataLoader(transformed_dataset, batch_size=4,
    #                    shuffle=True, num_workers=4)
