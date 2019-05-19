import pandas as pd
import requests

def test_download():
    dataframe = pd.read_csv('first_and_last_names.csv')
    dir = 'downloaded_images/'
    img_links = dataframe.iloc[:,:]
    img_numbers = list(range(0, 129000))

    for i,url in enumerate(img_links['image_url']):
        print('Downloading Image\t{0}'.format(img_numbers[i]))
        image=requests.get(url, stream=True)
        if image.status_code==200:
            with open(dir+'image_'+str(img_numbers[i])+'.jpg','wb')as f:
                for chunk in image.iter_content(1024):
                    f.write(chunk)



if __name__ == '__main__':
    test_download()