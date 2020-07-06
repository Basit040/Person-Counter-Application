#!/usr/bin/env python3
"""
Created on Sun May 17 08:10:45 2020

@author: Abdul Basit
"""
"""People Counter Application Using INTEL OPENVINO"""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
# Contains code for working with the Inference Engine


# Importing necessary libraries
import os
import sys
#Flexible Event Logging System
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        #Initialize desired class variables 
        self.network = None
        self.plugin = None
        self.exec_network = None
        self.input_blob = None
        self.out_blob = None
        self.infer_request = None


    def load_model(self,model,cpu_extension, device):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        
        # Load the model
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Initialize the plugin
        self.plugin = IECore()
        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        # Check for supported layers
        supported_layers = self.plugin.query_network(self.network,device)
        # Check unsupported layers
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        # When unsupported layer has been found
        if len(unsupported_layers) != 0:
            #log.error("Unsupported layers found: {}".format(unsupported_layers))
            #log.error("Check whether  cpu  extensions are available to add to IECore.")
            #sys.exit(1)
            #Above code for execute error when unsupported layers are found
            # Add any necessary extensions 
            self.plugin.add_extension(cpu_extension, device)
        # Return the loaded inference plugin 
        self.exec_network = self.plugin.load_network(self.network , device)
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return self.exec_network;

    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        # Return the shape of the input layer
        return self.network.inputs[self.input_blob].shape

    def exec_net(self,frame):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        # Start an asynchronous request
        self.exec_network.start_async(request_id=0,inputs={self.input_blob: frame})
        return self.exec_network

    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        # Wait for the request to be complete.
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        #Extract and return the output results
        return self.exec_network.requests[0].outputs[self.output_blob]