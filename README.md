## App for defected nails classification

A simple model and REST api for bad/good nail classification.  

More information in `Report.ipynb`.

### Installation 

```bash
$ pip install -r requirements.txt
$ python setup.py 
```

### Docker usage

```
docker image build -t nailgun .
docker run -t nailgun -d
curl http://<DOCKER-IP>:5000/predict?image_url=http://domain.com/image.jpeg
```