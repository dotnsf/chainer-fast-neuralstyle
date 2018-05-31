from __future__ import print_function
import numpy as np
import argparse
from PIL import Image, ImageFilter
import time

import chainer
from chainer import cuda, Variable, serializers
from net import *

import sys
import os
from flask import Flask, request, render_template, Response
from werkzeug.routing import BaseConverter
app = Flask(__name__)


class RegexConverter(BaseConverter):
  def __init__(self, url_map, *items):
    super( RegexConverter, self).__init__(url_map)
    self.regex = items[0]

app.url_map.converters['regex'] = RegexConverter


# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')

@app.route('/')
def index():
  name = "TEST"
  return render_template('index.html', title='chainer-gogh test', name=name)

@app.route('/models')
def models():
  files = os.listdir( 'models' );
  return Response( response=files, content_type='application/json' )

@app.route('/<regex("[0-9]*"):uid>.jpg')
def image(uid):
  #return 'tmp/' + uid + '.jpg'
  IMAGEFILE = 'tmp/output_%s.jpg' % (uid)
  f = open( IMAGEFILE, 'rb' )
  image = f.read()
  return Response( response=image, content_type='image/jpeg' )

@app.route('/post', methods = ['POST'])
def post():
  start = time.time()

  INPUTFILE = 'tmp/input_%d.jpg' % (start)
  OUTPUTFILE = 'tmp/output_%d.jpg' % (start)

  filebuf = request.files.get( 'image' )
  stream = filebuf.stream

  stylemodel = 'models/seurat.model'
  if request.form['stylemodel'] :
    stylemodel = 'models/%s.model' % request.form['stylemodel']

  f = open( INPUTFILE, 'wb' )
  f.write( stream.read() )

  # chainer
  model = FastStyleNet()
  serializers.load_npz(stylemodel, model)
  xp = np

  original = Image.open( INPUTFILE ).convert( 'RGB' )
  image = np.asarray( original, dtype=np.float32 ).transpose( 2, 0, 1 )
  image = image.reshape((1,) + image.shape)
  image = np.pad( image, [[0,0],[0,0],[50,50],[50,50]], 'symmetric')
  image = xp.asarray(image)

  x = Variable(image)
  y = model(x)
  result = cuda.to_cpu(y.data)
  result = result[:, :, 50:-50, 50:-50]
  result = np.uint8(result[0].transpose((1,2,0)))
  med = Image.fromarray(result)
  med = med.filter(ImageFilter.MedianFilter(3))
  #med = original_colors(original, med)
  end = time.time()
  print(end - start, 'sec')
  med.save( OUTPUTFILE )
  #os.remove( INPUTFILE )

  return '%d' % start


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=5000)
