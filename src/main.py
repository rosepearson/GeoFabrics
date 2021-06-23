# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
from HydrologicDEMs import processor
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--instructions', metavar='path', required=True, action='store',
                        help='the path to instruction file')
    
    return parser.parse_args()

def launch_processor(args):
    with open(args.instructions, 'r') as file_pointer:
        instructions = json.load(file_pointer)

    generator = processor.GeoFabricsGenerator(instructions)
    generator.run()

def main():
    args = parse_args()
    launch_processor(args)

if __name__ == "__main__":
    main()
        
        
