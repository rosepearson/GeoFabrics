# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:49 2021

@author: pearsonra
"""
from HydrologicDEMs import processor
import json

def main():
    with open(r'C:\Users\pearsonra\Documents\repos\Hydrologic-DEMs\instruction.json', 'r') as fp:
        instructions = json.load(fp)
        
    generator = processor.HydrologicalDemGenerator(instructions)
    generator.run()

if __name__ == "__main__":
    # execute only if run as a script
    main()
        
        
