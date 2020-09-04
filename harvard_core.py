# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2020 All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
# --------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from boaas_sdk import BOaaSClient

import argparse

"""
This example demonstrates BOA usage for a tabulated function stored as a table file.

The BOA SDK has been designed to be simple to use, but flexible in the range of
configurations available for tailoring the optimization. This is achieved using
the BOaaSClient object, which facilitates all communication with the BOA server.
The optimization configuration is handled via a Python dictionary (or
equivalently a JSON file).

"""

## Setup argparse for command-line inputs
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description = '''
        This example demonstrates basic BOA usage for a simple optimization problem.

        The BOA SDK has been designed to be simple to use, but flexible in the range of
        configurations available for tailoring the optimization. This is achieved using
        the BOaaSClient object, which facilitates all communication with the BOA server.
        The optimization configuration is handled via a Python dictionary (or
        equivalently a JSON file).
        ''')
argparser.add_argument('--hostname',
    dest    = 'clientHost',
    action  = 'store',
    default = 'localhost',
    help    = 'Set hostname to connect to. Defaults to "localhost"')

## Parse command-line arguments
args = argparser.parse_args()

hostname = 'http://{}:5000'.format(args.clientHost)
print ("Connecting to host: {}".format(hostname))
boaas = BOaaSClient(host=hostname)

# Read the 18-column data:
# config  benchmark  cycle  inst  power
# depth  width  gpr_phys  br_resv  dmem_lat  load_lat  br_lat  fix_lat  fpu_lat
# d2cache_lat  l2cache_size  icache_size  dcache_size
df = pd.read_table("input/data_model_ammp.txt")
#print(df)
# Get bips (billions of instructions per second)
# Ref: http://people.duke.edu/~bcl15/code/core/code_asplos06.txt
df['bips']=df['inst']/1.1*df['depth']/df['cycle']/18
domain = df[['depth',  'width', 'gpr_phys', 'br_resv', 'dmem_lat', 'load_lat', 'br_lat', 'fix_lat', 'fpu_lat', 'd2cache_lat', 'l2cache_size', 'icache_size', 'dcache_size']].to_numpy().tolist()

def myfunc(x):
    # 13-parameter func for core performance simulation
    depth = int(0.5 + x[0])
    width = int(0.5 + x[1])
    gpr_phys = int(0.5 + x[2])
    br_resv = int(0.5 + x[3])
    dmem_lat = int(0.5 + x[4])
    load_lat = int(0.5 + x[5])
    br_lat = int(0.5 + x[6])
    fix_lat = int(0.5 + x[7])
    fpu_lat = int(0.5 + x[8])
    d2cache_lat = int(0.5 + x[9])
    l2cache_size = int(0.5 + x[10])
    icache_size = int(0.5 + x[11])
    dcache_size = int(0.5 + x[12])

    # Get the function value from the corresponding row
    # df.loc[... is a series.   .iloc[0] is the first (only) element of the series - should have a better way
    bips = df.loc[(df.depth==depth) & (df.width==width) & (df.gpr_phys==gpr_phys) & (df.br_resv==br_resv) & (df.dmem_lat==dmem_lat) & (df.load_lat==load_lat) & (df.br_lat==br_lat) & (df.fix_lat==fix_lat) & (df.fpu_lat==fpu_lat) & (df.d2cache_lat==d2cache_lat) & (df.l2cache_size==l2cache_size) & (df.icache_size==icache_size) & (df.dcache_size==dcache_size), 'bips'].iloc[0]
    print("aaa000 LOCALS", depth,  width, gpr_phys, br_resv, dmem_lat, load_lat, br_lat, fix_lat, fpu_lat, d2cache_lat, l2cache_size, icache_size, dcache_size, bips)
    return bips

experiment_config = {
    "name": "Harvard core",
    "domain": domain,
    "model":{"gaussian_process": {
    "kernel_func": "Matern52",
    "scale_y": True,
    "scale_x": False,
    "noise_kernel": True,
    "use_scikit": True
     }},
    "optimization_type": "max",
    "initialization": {
      "type": "random",
      "random": {
        "no_samples": 3,
        "seed": None
      }
    },
    "sampling_function": {
    "type": "expected_improvement",
    "epsilon": 0.03,
    "optimize_acq": False,
    "outlier": False,
    "bounds": None
  }

}

user = {"_id": "boa_test@test.com", "password": "password"}
user_login = boaas.login(user)

if user_login == None:
    user = {"_id": "boa_test@test.com", "name": "BOA Test",
            "password": "password", "confirm_password": "password" }
    boaas.register(user)
    user_login = boaas.login(user)

print(user_login)
user_token = user_login["logged_in"]["token"]
print("user token")
print(user_token)
create_exp_user_object = { "_id": user["_id"], "token": user_token}
experiment_res = boaas.create_experiment(create_exp_user_object, experiment_config)
print(experiment_res)
experiment_id = experiment_res["experiment"]["_id"]
print("DBG-experiment_id = ", experiment_id)
boaas.run(experiment_id=experiment_id, user_token=user_token, func=myfunc, no_epochs=60, explain=False)
best_observation = boaas.best_observation(experiment_id, user_token)
print("best observation:")
print(best_observation)
boaas.stop_experiment(experiment_id=experiment_id, user_token=user_token)
