# Title: Estimate Distance Decay Function Parameters
# Author: Ryan McWay
# Description: For Estimap, we need an update of the k and alpha parameters.
#       To do this, for several counties, I use known travel decisions to estimate likelihood of traveling to a park given the distance of that park from their home. 
#       I need to do this for both recreation and tourism (e.g., domestic vs. international)


# Load necessary libraries
library(rio) # Open any data
library(data.table) # Fast data processing 

#####################
#   Recreation Estimations
#####################



# United Kingdom: Monitoring of Natural Environments
    # Use the MENE survey https://www.gov.uk/government/collections/monitor-of-engagement-with-the-natural-environment-survey-purpose-and-results
    # https://publications.naturalengland.org.uk/publication/2248731?category=47018

# TODO: I actually don't know how to access this data. 
# TODO: This provides number of visits in j from i. I will need the population of i to turn the outcome measure into a probability for the regression. 


# United States: Iowa Lake Project
    # https://lakes.card.iastate.edu/interactive-maps
    
# Would need to manually (or web scrap) record this into useable data from interactive maps
# For each lake, it shows percent of visitors by distance in bins. 
# This is for several years. So I would need to create this data for each year. v


# United States: National Parks
    # Socioeconomic Monitoring Visitor Surveys: https://www.nps.gov/subjects/socialscience/socioeconomic-monitoring-visitor-surveys.htm
    # https://www.nps.gov/subjects/socialscience/visitor-use-statistics-dashboard.htm
    # Query Builder for Historic Annual Recreation Visits (1904 - 1979)
    # Query Builder for Visitor Use Statistics (1979 - Last Calendar Year)

# TODO: I see how you can measure number of visits to site j, but I don't see where the visitors come from (place i)



# Switzerland: Microcensus on Mobility
    #  Swiss National Microcensus on mobility and transport
    # https://www.are.admin.ch/are/en/home/mobility/data/mtmc.html
    # Details on this study in Section 2: https://www.sciencedirect.com/science/article/pii/S0143622819302553#sec2
    
# TODO: Need Spencer to help me obtain a NDA to access the data. 
# "Once they have signed a non-disclosure agreement, both private and public research institutes are able to access the data in anonymised form (please see ‘Contact')"



# Italy: 1996 Survey
    # 1996 survey of outdoor recreation
    # https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1477-9552.2000.tb01225.x?saml_referrer=

# TODO: Would need to reach out to the authors about data access. 


# Italy: Natura survey 
    # Natura 2000 surveys: https://www.sciencedirect.com/science/article/pii/S2213078018300033
    # Don't think will be useful.
    
# TODO: Only record park j and if they are from the same province, region, italy, or abroad from the location. 
# So not telling us home location making the distance variable difficult to construct (Appendix Table A1).


# Netherlands: Continuous Free Time Survey (CVTO)
    # Paper uses CVTO: https://www.sciencedirect.com/science/article/pii/S0965856416302105?via%3Dihub

# TODO: Unclear how to access this data. 

# Finland: Survey
    # Paper: https://sci-hub.ru/https://doi.org/10.1080/15022250600658937

# TODO: Would need to reach out to the authors. 