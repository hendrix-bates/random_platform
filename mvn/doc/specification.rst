Maven Signal Project
========================

We have a project for you to work on. 
We'd like you to take a shot at creating a predictive signal with the data we are sharing with you. 
The goal is for you to use python for this exercise and provide us a brief write-up of the work 
you have done in addition to the code you write. 
Some analysis of why this signal is good in your view and how you analysed the quality of the signal. 

The broader scoring criteria is below.
The goal is to create something which predicts returns (either the 'ret' or 'resRet2' fields).   
This signal should have predictive power for a minimum of 2 forward time bins, 
but ideally a bit longer as one thinks about potential trading considerations.

If you need to use cloud computational resource, 
you can use for free here  https://colab.research.google.com/notebooks/intro.ipynb   

Scoring criteria
------------------

- Code Structure
- Code quality
- Presentation and Write-up
- Lookahead and Survivorship bias
- Feature engineering (alpha/signal creation) and analysis
- Justification for approach taken
- Consideration of multivariate modeling techniques (use, reasoning, applicability, etc)
- Analysis of results
- Practical considerations, i.e. implementation, trading, risk
- Suggestions and considerations for further work/improvements

The data can be found at: https://drive.google.com/drive/folders/1VGg2EY1S7j9c8IG7w4P2mUSUM9CFtzHE 

It can be read into python as a dictionary of arrays, with the below keys.
The data is saved in NxS matrices where N is the number of time bins, while S is the number of unique securities in the matrix where each column represents a unique security.

Field descriptions:

- date - integer indicating ordering of the rows of time bins
- opn - open price for the time bin, unadjusted, local
- clo - close price for the time bin , unadjusted, local
- ret - security return for the time bin
- volume - volume of shares traded for the time bin
- evtMat - integer -20 to 20, where negative numbers indicate the number of time bins until we expect an event to occur, 0 is the time bin within which an event has occurred, positive integers are the number of days since we think an event occured
- filter - binary 1,0...1s indicate securities that should be considered as part of the estimation universe without lookahead or survivorship bias.  I would only use securities that are in the estimation universe for modeling to avoid bias in the model.
- liqScore - liquidity score, indicating how liquid a stock is
- mktCap - market cap of a stock in USD
- modelBeta - estimated stock beta
- resRet2 - residualized stock return for a bin, this is generally what we want to estimate
- b* fields (i.e. bB2P, bBetaNL, ...) - style factor exposures
- capBands - bins for stocks as broken up by mktCap
- adv30 - 30 bin median stock average value traded in USD
- adjSpl - split adjustment factor
- adjSplClo  - split adjusted price for close of bin in USD
- ctryId - integer indicating what country the security trades in 
- ind1,2,3 - integer indicating industry group the stock belongs to
- stMnCcFact - pls ignore

fields:
    {'date'      }
    {'opn'       }
    {'clo'       }
    {'ret'       }
    {'volume'    }
    {'evtMat'    }
    {'filter'    }
    {'liqScore'  }
    {'mktCap'    }
    {'modelBeta' }
    {'resRet2'   }
    {'bB2P'      }
    {'bBetaNL'   }
    {'bDivYld'   }
    {'bErnYld'   }
    {'bLev'      }
    {'bLiquidity'}
    {'bMomentum' }
    {'bResVol'   }
    {'bSize'     }
    {'bSizeNL'   }
    {'capBands'  }
    {'adv30'     }
    {'adjSpl'    }
    {'stMnCcFact'}
    {'adjSplClo' }
    {'ctryId'    }
    {'ind1'      }
    {'ind2'      }
    {'ind3'      }  

code snippet to assist in opening the file.

.. code-block:: python

    import h5py
    fname = 'Z:\\QuantEquity\\coop\\share\\Intern\\Sept2019\\sampleData_US.mat'
    def read_h5(fname):
        f = h5py.File(fname, 'r')
        dct={}
        for k,v in f.items():
            dct[k] = v.value[0]
        return dct

    dct = read_h5(fname)
    df = pd.DataFrame(dct) 