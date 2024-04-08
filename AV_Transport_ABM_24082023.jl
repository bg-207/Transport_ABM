using Agents
using Random
using CSV
using Distributions
using DataFrames
using Plots
using GLMakie

size = (20, 20)
space = GridSpaceSingle(size; periodic = false, metric = :chebyshev)


# Right-skewed distribution
function random_human_age()
    age = Int(trunc(exp(rand(Normal(log(30), 1.5)))))
    return age > 120 ? 120 : age
end

function random_australian_age()
    # Assign probabilities for each age group based on ABS data
    # https://www.abs.gov.au/statistics/people/population/national-state-and-territory-population/sep-2021
    v = [0.025, 0.028, 0.028, 0.031, 0.032, 0.033, 0.032, 0.033, 0.035, 0.037, 0.039, 0.042, 0.043, 0.045, 0.047, 0.049, 0.051, 0.054, 0.054, 0.053, 0.046, 0.036, 0.027, 0.015, 0.005]
    age_groups = collect(0:4:120)

    s = sum(v)
    probs = v ./ s

    age_group = rand(Categorical(probs))
    age = rand(age_groups[age_group]:age_groups[age_group]+4)
    return trunc(Int, age)
end

# 1 = Male, -1 = Female
function random_human_gender()
    gender = rand(Bernoulli(0.5)) == 1 ? 1 : 0
    return gender
end

function random_australian_education(age::Int)
    if age < 15
        return 1 # No educational qualification
    elseif age < 20
        v = [0.536, 0.248, 0.086, 0.064, 0.035, 0.031]
    elseif age < 25
        v = [0.232, 0.308, 0.190, 0.090, 0.095, 0.085]
    elseif age < 35
        v = [0.072, 0.104, 0.350, 0.181, 0.139, 0.155]
    elseif age < 45
        v = [0.058, 0.060, 0.305, 0.230, 0.169, 0.178]
    elseif age < 55
        v = [0.042, 0.035, 0.274, 0.254, 0.169, 0.226]
    elseif age < 65
        v = [0.039, 0.029, 0.219, 0.284, 0.172, 0.257]
    else
        v = [0.848, 0.021, 0.042, 0.029, 0.030, 0.030]
    end
    
    s = sum(v)
    probs = v ./ s

    # Generate a random education level based on the probabilities
    education = rand(Categorical(probs))

    # Return the education level as a category number
    # 1: No educational qualification
    # 2: Primary education
    # 3: Secondary education
    # 4: Certificate level
    # 5: Advanced diploma or diploma
    # 6: Bachelor's degree or higher
    return education
end

function random_australian_employment(age)
    # Assign probabilities for each employment status based on ABS data
    # https://www.abs.gov.au/statistics/labour/employment-and-unemployment/labour-force-australia/latest-release
    if age >= 65
        v = [0.070, 0.023, 0.510, 0.064, 0.315, 0.018, 0.000]
    elseif age >= 25
        v = [0.692, 0.051, 0.050, 0.022, 0.100, 0.084, 0.001]
    else
        v = [0.721, 0.048, 0.010, 0.027, 0.097, 0.092, 0.005]
    end
    
    s = sum(v)
    probs = v ./ s

    # Generate a random employment status based on the probabilities
    employment = rand(Categorical(probs))

    # Return the employment status as a category number
    # 1: Employed - full time
    # 2: Employed - part time
    # 3: Unemployed - actively looking for work
    # 4: Unemployed - not actively looking for work
    # 5: Not in the labour force - retired
    # 6: Not in the labour force - studying or training
    # 7: Not in the labour force - other reasons
    return employment
end

function generate_income(age::Int, gender::Int, education::Int, employment::Int)
    # Calculate base income based on age and gender
    if age < 15
        base_income = 10000 # Invalid age
    elseif age < 20
        base_income = 20000 + 5000 * gender # Age 15-19
    elseif age < 25
        base_income = 30000 + 5000 * gender # Age 20-24
    elseif age < 35
        base_income = 50000 + 10000 * gender # Age 25-34
    elseif age < 45
        base_income = 65000 + 10000 * gender # Age 35-44
    elseif age < 55
        base_income = 75000 + 10000 * gender # Age 45-54
    elseif age < 65
        base_income = 75000 + 5000 * gender # Age 55-64
    else
        base_income = 50000 # Age 65-85
    end
    
    # Adjust income based on education level
    if education < 4
        base_income *= 0.7 # No post-school qualification
    elseif education <6
        base_income *= 1.0 # Post-school qualification below bachelor level
    elseif education == 6
        base_income *= 1.3 # Bachelor degree or above
    else
        base_income = 30000 # Invalid education level
    end
    
    # Adjust income based on employment status
    if employment == 1
        base_income *= 1.2 # Full-time
    elseif employment == 2
        base_income *= 0.8 # Part-time
    elseif employment == 3 || employment == 4
        base_income *= 0.5 # Unemployed
    elseif employment > 4
        base_income *= 0.6 # Not in the labour force
    else
        base_income = 30000 # Invalid employment status
    end
    
    # Add random noise to income
    noise = rand(Normal(0, 0.2 * base_income)) # 20% standard deviation
    income = round(base_income + noise)
    income = trunc(Int, income)
    
    # Return income as integer
    return max(income, 0) # Ensure income is non-negative
end

# Caloric Intake and Incidental Exercise
function random_caloric_intake(gender::Int)
    if gender == 1
        intake = clamp(rand(Normal(8800,1500)), 2000, 15000)
    else
        intake = clamp(rand(Normal(6500,1200)), 2000, 12000)
    end
    return intake
end

function random_incidental_exercise()
    μ = 837  # mean daily incidental exercise in kJ
    σ = 150  # standard deviation of daily incidental exercise in kJ
    return clamp(rand(Normal(μ, σ)), 0, 2000)
end

# Generate random positions in the grid
function generate_random_position(grid::GridSpace)
    x = rand(1:size(grid)[1])
    y = rand(1:size(grid)[2])
    return (x, y)
end

function transport_probability()
    rand_num = rand()
    if rand_num < 0.58
        return 2 # Private car
    elseif rand_num < 0.82
        return 6 # Walking
    elseif rand_num < 0.94
        return 3 # Public transport
    elseif rand_num < 0.96
        return 5 # Cycling
    elseif rand_num < 0.98
        return 7 # Automated Ride-hail
    elseif rand_num < 0.99
        return 4 # Personal micromobility
    else
        return 8 # Car sharing
    end
end


@agent struct TransportAgent(GridAgent{2}) 
    #DEMOGRAPHICS 
    age::Int
    gender::Int
    education::Int
    employment::Int
    income::Int

    #TRANSPORT LAYER
    original_transport_type ::Int64 # 2 = Car, 3 = Public Transport, 4 = Personal micromobility,  5 = Cycling, 6 = Walking, and 7 = ride-hail, 8 = car sharing, BUT ALL START WITH 0. 
    transport_type::Int64 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Personal micromobility,  5 = Cycling, 6 = Walking, and 7 = ride-hail, 8 = car sharing, 9 = AV ride hail 
    transport_choice::Int64 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Personal micromobility,  5 = Cycling, 6 = Walking, and 7 = ride-hail, 8 = car sharing, 9 = AV ride hail 

    #COGNITIVE LAYER - AUTONOMOUS VEHICLES
    #Theory of planned behaviour
    av_attitudes::Float32
    av_social_norms::Float32
    av_control_factors::Float32
    av_behavioural_intention::Int64
    av_subjective_norm::Float32
    av_facilitating_conditions::Float32
    av_threshold::Float32

    #COGNITIVE LAYER - AUTONOMOUS RIDE HAIL APPS
    #Theory of planned behaviour
    rh_attitudes::Float32
    rh_social_norms::Float32
    rh_control_factors::Float32
    rh_behavioural_intention::Int64
    rh_subjective_norm::Float32
    rh_facilitating_conditions::Float32
    rh_threshold::Float32
    rh_fee_applied::Bool

    # Public transport layer 
    near_public_transport::Bool

    #Additional cognitive factors 
    impulsivity::Float32

    # Cognitive behavioural control - AV
    av_cb_pos::Float32
    av_cb_neg::Float32

    # Cognitive behavioural control - RH
    rh_cb_pos::Float32
    rh_cb_neg::Float32

    # PHYSICAL HEALTH LAYER
    physical_health_layer::Float32
    sedentary_behaviour::Float32

    # PUBLIC TRANSPORT AGENTS 
    is_pt_agent::Bool
    pt_coverage_radius::Int 
    fee::Int 

    # PROMOTION AGENT 
    is_promotion_agent::Bool
    promotion_coverage_radius::Int
    av_advertising_efficacy::Int
    pt_active_transport_advertising_efficacy::Float32

    # COGNITIVE LAYER FOR WALKING 
    #Theory of planned behaviour
    walking_attitudes::Float32
    walking_social_norms::Float32
    walking_control_factors::Float32
    walking_behavioural_intention:: Float32
    walking_subjective_norm::Float32
    walking_facilitating_conditions::Float32
    walking_threshold::Float32

    #COGNITIVE LAYER - CYCLING
    #Theory of planned behaviour
    cycling_attitudes::Float32
    cycling_social_norms::Float32
    cycling_control_factors::Float32
    cycling_behavioural_intention::Float32
    cycling_subjective_norm::Float32
    cycling_facilitating_conditions::Float32
    cycling_threshold::Float32

    # Control behaviours - Walking
    walking_cb_pos::Float32
    walking_cb_neg::Float32

    # Control behaviours - Cycling
    cycling_cb_pos::Float32
    cycling_cb_neg::Float32

end

function agent_step!(agent, model)
    update_near_public_transport(agent, model)
    consolidated_transport_decision!(agent, model)
    agent_health!(agent, model)
    # apply_rebate_after_purchase!(agent, model)
end

function model_step!(model)
    model.tick += 1

    if model.tick <= 90
        model.av_threshold_model = 2
        model.rh_threshold_model = 2
    
    elseif model.tick > 90 && model.tick <= 180
        model.av_threshold_model = 1.7
        model.rh_threshold_model = 1.7

    elseif model.tick > 180 && model.tick <= 270
        model.av_threshold_model = 1.5
        model.rh_threshold_model = 1.5
    
    elseif model.tick > 270 && model.tick <= 360
        model.av_threshold_model = 1.2
        model.rh_threshold_model = 1.2

    else 
        model.av_threshold_model = 1
        model.rh_threshold_model = 1
    end

end

# Assign whether the agents are near public transport

function update_near_public_transport(agent, model)
    for pt_agent in nearby_agents(agent, model, 3)
        if pt_agent.is_pt_agent == true
            agent.near_public_transport = true 
            return
        end
    end
end



# AV DECISION-MAKING USING THE THEORY OF PLANNED BEHAVIOUR # 
function AV_TPB(av_attitudes, av_control_factors, av_social_norms, av_subjective_norm, av_faciliating_conditions, threshold)
    if false
        print("Facil Conditions: ", av_faciliating_conditions, " Sum: ", sum(av_attitudes)+sum(av_control_factors)+sum(av_social_norms)+sum(av_subjective_norm),
            " Threshold: ", threshold, "\n")
    end
    return all(av_faciliating_conditions) &&
        sum(av_attitudes)+
        sum(av_control_factors)+
        sum(av_social_norms)+
        sum(av_subjective_norm) > model.av_threshold_model
        print(return)
end

# RIDE-HAIL DECISION-MAKING USING THE THEORY OF PLANNED BEHAVIOUR # 

function RH_TPB(rh_attitudes, rh_control_factors, rh_social_norms, rh_subjective_norm, rh_faciliating_conditions, threshold)
    if false
        print("Facil Conditions: ", rh_faciliating_conditions, " Sum: ", sum(rh_attitudes)+sum(rh_control_factors)+sum(rh_social_norms)+sum(rh_subjective_norms),
            " Threshold: ", threshold, "\n")
    end
    return all(rh_faciliating_conditions) &&
        sum(rh_attitudes)+
        sum(rh_control_factors)+
        sum(rh_social_norms)+
        sum(rh_subjective_norm) > model.rh_threshold_model
    print(return)
end



# POLICIES: ASSIGNING FEES FOR SHORT RIDE-HAIL TRIPS AND TRIPS WHERE PUBLIC TRANSPORT IS NEARBY 
# If the randomly generated trip for a ride-hail is a short distance, then there will be a 50% increase in price. 
function assign_rh_trip_cost(agent_trip_distance, agent, model)

    trip_cost = model.rh_trip_cost # start with the default cost

    # FEES FOR SHORT TRIPS 
    short_trip_threshold = 5 # You can adjust this value.
    short_trip_fee_amount = 5 # Example fee amount for short trips.

    if agent_trip_distance <= short_trip_threshold
        trip_cost += short_trip_fee_amount
    end

    # FEES FOR TRIPS WHERE PUBLIC TRANSPORT IS NEARBY 
    nearby_public_transport_fee = 10
    if agent.near_public_transport
        trip_cost += nearby_public_transport_fee
    end

    # print("Price: $trip_cost")
    return trip_cost
end

# POLICY: VISIBLE REBATE FOR AGENTS
# This rebate changes the price of the AV so that agents can see it decrease, essentially appearing as a 'discount' for private AVs. 

AV_rebate_full_amount = 5000

function apply_rebate!(agent, model, rebate_amount)

    private_av_price = model.private_AV_cost - rebate_amount # Reduce the cost of private AV by the rebate amount

    return private_av_price
end

# POLICY: PROMOTION OF AUTONOMOUS VEHICLES 

function av_promotion_policy!(agent, model)
    for av_promotion_agent in nearby_agents(agent, model, 3)
        if av_promotion_agent.is_promotion_agent == true
            agent.av_attitudes + av_promotion_agent.av_advertising_efficacy 
        end
    end
end


function consolidated_transport_decision!(agent, model)
    
    if agent.is_pt_agent == false && agent.is_promotion_agent == false

        # Calculate AV Decision
        # POLICY OPTION: AV PROMOTION VIA ADVERTISING 
        #av_promotion_policy!(agent, model)
        
        av_attitudes = agent.av_attitudes
        av_control_behaviour = agent.av_cb_pos - agent.av_cb_neg
    #     # calculating social_norms
        av_subjective_norms = 0
        av_num_neighbors = 0
        for av_neighbor in nearby_agents(agent, model)
            if av_neighbor.is_pt_agent == false && av_neighbor.is_promotion_agent == false
                agent_av_subjective_norms = av_subjective_norms + av_neighbor.av_attitudes
                av_num_neighbors = av_num_neighbors + 1
            end
        end
        
        #Taking the average
        av_subjective_norms = av_num_neighbors==0 ? 0 : av_subjective_norms / av_num_neighbors    

        av_descriptive_norms = model.AVs / model.total_agents

        # COMMENT THIS OUT TO TURN OFF THE REBATE POLICY
        #AVs not implemented yet code:
        # if model.tick >= 200
        #     av_facil_conditions = [agent.income > apply_rebate!(agent, model, AV_rebate_full_amount)]
        # else
        #     av_facil_conditions = [agent.income > model.private_AV_cost]
        # end

        # AVs implemented from the start code with rebate policy:
        #av_facil_conditions = [agent.income > apply_rebate!(agent, model, AV_rebate_full_amount)]

        # NO REBATE POLICY
        av_facil_conditions = [agent.income > model.private_AV_cost]

        av_decision = AV_TPB(av_attitudes, av_control_behaviour, av_subjective_norms, av_descriptive_norms, av_facil_conditions, model.av_threshold_model)

        # Calculate Ride-Hail Decision
        rh_attitudes = agent.rh_attitudes
        rh_control_behaviour = agent.rh_cb_pos - agent.rh_cb_neg
        rh_descriptive_norms = model.RH_trips / model.total_agents
        # calculating social_norms
        rh_subjective_norms = 0
        rh_num_neighbors = 0
        for rh_neighbor in nearby_agents(agent, model)
            if rh_neighbor.is_pt_agent == false && rh_neighbor.is_promotion_agent == false
                agent_rh_subjective_norms = rh_subjective_norms + rh_neighbor.rh_attitudes
                rh_num_neighbors = rh_num_neighbors + 1
            end
        end
        
        #Taking the average
        rh_subjective_norms = rh_num_neighbors==0 ? 0 : rh_subjective_norms / rh_num_neighbors   
        agent_trip_distance = rand(1:20)
        # IF FEES FOR SHORT TRIPS AND FOR NEARBY PUBLIC TRANSPORT ARE BEING IMPLEMENTED, ACTIVATE CODE BELOW:
        # Implementation of short trips and nearby public transport fees policies after x steps:
        # if model.tick >= 150
        #     rh_facil_conditions = [(agent.income*0.0005) > assign_rh_trip_cost(agent_trip_distance, agent, model)] 
        # else
        #     rh_facil_conditions = [(agent.income*0.0005) > model.rh_trip_cost] 
        # end

        # If fees policies for RH are applied from the start: 

        #rh_facil_conditions = [(agent.income*0.0005) > assign_rh_trip_cost(agent_trip_distance, agent, model)] 

        # IF FEES FOR SHORT TRIPS POLICY IS NOT BEING IMPLEMENTED, ACTIVATE CODE BELOW: 
        rh_facil_conditions = [(agent.income*0.001) > model.rh_trip_cost]
        rh_decision = RH_TPB(rh_attitudes, rh_control_behaviour, rh_subjective_norms, rh_descriptive_norms, rh_facil_conditions, model.rh_threshold_model)


        
        # Final Decision
        # This is an example of a decision hierarchy - you can adjust as required.
        if av_decision 
            agent.transport_choice = 1
            model.AVs += 1
            push!(model.AVs_time_series, model.AVs_time_series[end] + 1)
        elseif rh_decision
            agent.transport_choice = 9
            push!(model.RH_trips_time_series, model.RH_trips_time_series[end] + 1)
            model.RH_trips += 1
        else
            agent.transport_choice = agent.original_transport_type
        end
    end
end


# ORIGINAL AGENT HEALTH CODE 

# function agent_health!(agent, model) # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Cycling,  5 = Walking, 6 = Ride-hail app, BUT ALL START WITH 0. 
#     if agent.transport_choice == 1 || agent.transport_choice == 2 || agent.transport_choice == 6
#         if agent.original_transport_type == 2
#             agent.sedentary_behaviour = agent.sedentary_behaviour
#         else 
#             agent.sedentary_behaviour += 0.1
#             agent.physical_health_layer -= 0.01
#         end
#     end
# end

function agent_health!(agent, model) # 2 = Car, 3 = Public Transport, 4 = Personal micromobility,  5 = Cycling, 6 = Walking, and 7 = ride-hail, 8 = car sharing, BUT ALL START WITH 0. 
    if agent.original_transport_type == 3 || agent.transport_choice == 4 || agent.transport_choice == 5 || agent.transport_choice == 6
        if agent.transport_choice == 1 || agent.transport_choice == 2 || agent.transport_choice == 9 || agent.transport_choice == 7 || agent.transport_choice == 8
            agent.sedentary_behaviour += 1
            agent.physical_health_layer -= 1
        end
    end
end

properties = Dict(:private_AV_cost => 50000, :rh_trip_cost => 10, :tick => 1, :av_threshold_model => 5.0, :rh_threshold_model => 5.0, :AVs => 0, :RH_trips => 0, :AVs_time_series => [0], :RH_trips_time_series => [0], :total_agents => 250, :rh_fee_applied => false, :num_public_transport_agents => 100, :num_promotion_agents => 50)

using Random: MersenneTwister

model = StandardABM(
    TransportAgent, space; properties, 
    agent_step! = agent_step!, 
    model_step! = model_step!, 
    rng = MersenneTwister(100)
)


function initialize(; total_agents = 250, gridsize = (20, 20), private_AV_cost = 50000, rh_trip_cost = 10,  av_threshold_model = 5.0, rh_threshold_model = 5.0, AVs = 0, RH_trips = 0, AVs_time_series = [0], # Starting with 0 AVs
    RH_trips_time_series = [0], rh_fee_applied = false, num_public_transport_agents = 100, num_promotion_agents = 50, seed = 100)
    space = GridSpace(gridsize, periodic = false)
    properties = Dict(:private_AV_cost => private_AV_cost, :rh_trip_cost => rh_trip_cost, :tick => 1, :av_threshold_model => av_threshold_model, :rh_threshold_model => rh_threshold_model, :AVs => 0, :RH_trips => 0, :AVs_time_series => [0], :RH_trips_time_series => [0], :total_agents => total_agents, :rh_fee_applied => false, :num_public_transport_agents => 100, :num_promotion_agents => 50)
    rng = MersenneTwister(seed)
    model = StandardABM(TransportAgent, space; agent_step! = agent_step!, model_step! = model_step!,
    properties, rng, container = Vector, scheduler = Schedulers.Randomly())

    # Adding the Agents

    # Adding modal choice-making agents
    for n in 1:model.total_agents

        #DEMOGRAPHICS
        age = random_human_age()
        gender = random_human_gender()
        education = random_australian_education(age)
        employment = random_australian_employment(age)
        income = generate_income(age, gender, education, employment)

        #TRANSPORT LAYER
        original_transport_type = transport_probability() # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Personal micromobility,  5 = Cycling, 6 = Walking, and 7 = Automated ride-hail
        transport_type = 0 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Personal micromobility,  5 = Walking BUT ALL START WITH 0. 
        transport_choice = 0 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Personal micromobility,  5 = Walking, 6 = Ride-hail app, BUT ALL START WITH 0. 

        #COGNITIVE LAYER - AUTONOMOUS VEHICLES
        #Theory of planned behaviour
        av_attitudes = rand(abmrng(model))
        av_social_norms = rand(abmrng(model))
        av_control_factors = rand(abmrng(model))
        av_behavioural_intention = 0
        av_subjective_norm = rand(abmrng(model))
        av_facilitating_conditions = rand(abmrng(model))
        av_threshold = rand(abmrng(model))

        #COGNITIVE LAYER - AUTONOMOUS RIDE-HAIL APPS
        #Theory of planned behaviour
        rh_attitudes = rand(abmrng(model))
        rh_social_norms = rand(abmrng(model))
        rh_control_factors = rand(abmrng(model))
        rh_behavioural_intention = 0
        rh_subjective_norm = rand(abmrng(model))
        rh_facilitating_conditions = rand(abmrng(model))
        rh_threshold = rand(abmrng(model))
        rh_fee_applied = false

        #Public transport layer
        near_public_transport = false

        #Additional cognitive factors 
        impulsivity = rand(abmrng(model))

        # Control behaviours - AV 
        av_cb_pos = rand(abmrng(model))
        av_cb_neg = rand(abmrng(model))

        # Control behaviours - RH
        rh_cb_pos = rand(abmrng(model))
        rh_cb_neg = rand(abmrng(model))

        # PHYSICAL HEALTH LAYER
        physical_health_layer = rand(abmrng(model))
        sedentary_behaviour = rand(abmrng(model))

        # PUBLIC TRANSPORT AGENTS 
        is_pt_agent = false
        pt_coverage_radius = 0 
        fee = 0

        #  PROMOTION AGENT 
        is_promotion_agent = false 
        promotion_coverage_radius = 0
        av_advertising_efficacy = 0
        pt_active_transport_advertising_efficacy = 0

        #COGNITIVE LAYER - WALKING
        #Theory of planned behaviour
        walking_attitudes = rand(abmrng(model))
        walking_social_norms = rand(abmrng(model))
        walking_control_factors = rand(abmrng(model))
        walking_behavioural_intention = 0
        walking_subjective_norm = rand(abmrng(model))
        walking_facilitating_conditions = rand(abmrng(model))
        walking_threshold = rand(abmrng(model))

        #COGNITIVE LAYER - CYCLING
        #Theory of planned behaviour
        cycling_attitudes = rand(abmrng(model))
        cycling_social_norms = rand(abmrng(model))
        cycling_control_factors = rand(abmrng(model))
        cycling_behavioural_intention = 0
        cycling_subjective_norm = rand(abmrng(model))
        cycling_facilitating_conditions = rand(abmrng(model))
        cycling_threshold = rand(abmrng(model))

        # Control behaviours - Walking
        walking_cb_pos = rand(abmrng(model))
        walking_cb_neg = rand(abmrng(model))

        # Control behaviours - Cycling
        cycling_cb_pos = rand(abmrng(model))
        cycling_cb_neg = rand(abmrng(model))



        add_agent!(
            TransportAgent, model, age, gender, education, employment, income,
            original_transport_type, transport_type, transport_choice,
            av_attitudes, av_social_norms, av_control_factors, av_behavioural_intention, av_subjective_norm, av_facilitating_conditions, av_threshold,
            rh_attitudes, rh_social_norms, rh_control_factors, rh_behavioural_intention, rh_subjective_norm, rh_facilitating_conditions, rh_threshold, rh_fee_applied,
            near_public_transport, impulsivity, av_cb_pos, av_cb_neg, rh_cb_pos, rh_cb_neg,
            physical_health_layer, sedentary_behaviour,
            is_pt_agent, pt_coverage_radius, fee, is_promotion_agent, promotion_coverage_radius, av_advertising_efficacy, pt_active_transport_advertising_efficacy,
            walking_attitudes, walking_social_norms, walking_control_factors, walking_behavioural_intention, walking_subjective_norm, walking_facilitating_conditions, walking_threshold,
            cycling_attitudes, cycling_social_norms, cycling_control_factors, cycling_behavioural_intention, cycling_subjective_norm, cycling_facilitating_conditions, cycling_threshold,
            walking_cb_pos, walking_cb_neg, cycling_cb_pos, cycling_cb_neg
        )
    end

    # PROMOTION AGENTS 

    for _ in 1:model.num_promotion_agents
        #DEMOGRAPHICS
        age = 0
        gender = 0
        education = 0
        employment = 0
        income = 0

        #TRANSPORT LAYER
        original_transport_type = 0 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Personal micromobility,  5 = Walking, 6 = Ride-hail app, BUT ALL START WITH 0.
        transport_type = 0 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Personal micromobility,  5 = Walking, 6 = Ride-hail app BUT ALL START WITH 0. 
        transport_choice = 0 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Personal micromobility,  5 = Walking, 6 = Ride-hail app, BUT ALL START WITH 0. 

        #COGNITIVE LAYER - AUTONOMOUS VEHICLES
        #Theory of planned behaviour
        av_attitudes = 0
        av_social_norms = 0
        av_control_factors = 0
        av_behavioural_intention = 0
        av_subjective_norm = 0
        av_facilitating_conditions = 0
        av_threshold = 0

        #COGNITIVE LAYER - AUTONOMOUS RIDE-HAIL APPS
        #Theory of planned behaviour
        rh_attitudes = 0
        rh_social_norms = 0
        rh_control_factors = 0
        rh_behavioural_intention = 0
        rh_subjective_norm = 0
        rh_facilitating_conditions = 0
        rh_threshold = 0
        rh_fee_applied = false

        #Public transport layer
        near_public_transport = false

        #Additional cognitive factors 
        impulsivity = 0

        # Control behaviours - AV 
        av_cb_pos = 0
        av_cb_neg = 0

        # Control behaviours - RH
        rh_cb_pos = 0
        rh_cb_neg = 0

        # PHYSICAL HEALTH LAYER
        physical_health_layer = 0
        sedentary_behaviour = 0


        # PUBLIC TRANSPORT AGENTS 
        is_pt_agent = false
        pt_coverage_radius = 0 
        fee = 0

        #  PROMOTION AGENT 
        is_promotion_agent = true 
        promotion_coverage_radius = 2
        av_advertising_efficacy = 1
        pt_active_transport_advertising_efficacy = 1.5

        #COGNITIVE LAYER - WALKING
        #Theory of planned behaviour
        walking_attitudes = 0
        walking_social_norms = 0
        walking_control_factors = 0
        walking_behavioural_intention = 0
        walking_subjective_norm = 0
        walking_facilitating_conditions = 0
        walking_threshold = 0

        #COGNITIVE LAYER - CYCLING
        #Theory of planned behaviour
        cycling_attitudes = 0
        cycling_social_norms = 0
        cycling_control_factors = 0
        cycling_behavioural_intention = 0
        cycling_subjective_norm = 0
        cycling_facilitating_conditions = 0
        cycling_threshold = 0

        # Control behaviours - Walking
        walking_cb_pos = 0
        walking_cb_neg = 0

        # Control behaviours - Cycling
        cycling_cb_pos = 0
        cycling_cb_neg = 0

        add_agent!(
            TransportAgent, model, age, gender, education, employment, income,
            original_transport_type, transport_type, transport_choice,
            av_attitudes, av_social_norms, av_control_factors, av_behavioural_intention, av_subjective_norm, av_facilitating_conditions, av_threshold,
            rh_attitudes, rh_social_norms, rh_control_factors, rh_behavioural_intention, rh_subjective_norm, rh_facilitating_conditions, rh_threshold, rh_fee_applied,
            near_public_transport, impulsivity, av_cb_pos, av_cb_neg, rh_cb_pos, rh_cb_neg,
            physical_health_layer, sedentary_behaviour,
            is_pt_agent, pt_coverage_radius, fee, is_promotion_agent, promotion_coverage_radius, av_advertising_efficacy, pt_active_transport_advertising_efficacy,
            walking_attitudes, walking_social_norms, walking_control_factors, walking_behavioural_intention, walking_subjective_norm, walking_facilitating_conditions, walking_threshold,
            cycling_attitudes, cycling_social_norms, cycling_control_factors, cycling_behavioural_intention, cycling_subjective_norm, cycling_facilitating_conditions, cycling_threshold,
            walking_cb_pos, walking_cb_neg, cycling_cb_pos, cycling_cb_neg
        )
    end 

# PUBLIC TRANSPORT AGENTS

for _ in 1:model.num_public_transport_agents
    #DEMOGRAPHICS
    age = 0
    gender = 0
    education = 0
    employment = 0
    income = 0

    #TRANSPORT LAYER
    original_transport_type = 0 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Personal micromobility,  5 = Walking, 6 = Ride-hail app, BUT ALL START WITH 0.
    transport_type = 0 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Personal micromobility,  5 = Walking, 6 = Ride-hail app BUT ALL START WITH 0. 
    transport_choice = 0 # 1 = AV, 2 = Car, 3 = Public Transport, 4 = Personal micromobility,  5 = Walking, 6 = Ride-hail app, BUT ALL START WITH 0. 

    #COGNITIVE LAYER - AUTONOMOUS VEHICLES
    #Theory of planned behaviour
    av_attitudes = 0
    av_social_norms = 0
    av_control_factors = 0
    av_behavioural_intention = 0
    av_subjective_norm = 0
    av_facilitating_conditions = 0
    av_threshold = 0

    #COGNITIVE LAYER - AUTONOMOUS RIDE-HAIL APPS
    #Theory of planned behaviour
    rh_attitudes = 0
    rh_social_norms = 0
    rh_control_factors = 0
    rh_behavioural_intention = 0
    rh_subjective_norm = 0
    rh_facilitating_conditions = 0
    rh_threshold = 0
    rh_fee_applied = false

    #Public transport layer
    near_public_transport = false

    #Additional cognitive factors 
    impulsivity = 0

    # Control behaviours - AV 
    av_cb_pos = 0
    av_cb_neg = 0

    # Control behaviours - RH
    rh_cb_pos = 0
    rh_cb_neg = 0

    # PHYSICAL HEALTH LAYER
    physical_health_layer = 0
    sedentary_behaviour = 0


    # PUBLIC TRANSPORT AGENTS 
    is_pt_agent = true
    pt_coverage_radius = 2 
    fee = 0

    #  PROMOTION AGENT 
    is_promotion_agent = false 
    promotion_coverage_radius = 2
    av_advertising_efficacy = 1
    pt_active_transport_advertising_efficacy = 1.5

    #COGNITIVE LAYER - WALKING
    #Theory of planned behaviour
    walking_attitudes = 0
    walking_social_norms = 0
    walking_control_factors = 0
    walking_behavioural_intention = 0
    walking_subjective_norm = 0
    walking_facilitating_conditions = 0
    walking_threshold = 0

    #COGNITIVE LAYER - CYCLING
    #Theory of planned behaviour
    cycling_attitudes = 0
    cycling_social_norms = 0
    cycling_control_factors = 0
    cycling_behavioural_intention = 0
    cycling_subjective_norm = 0
    cycling_facilitating_conditions = 0
    cycling_threshold = 0

    # Control behaviours - Walking
    walking_cb_pos = 0
    walking_cb_neg = 0

    # Control behaviours - Cycling
    cycling_cb_pos = 0
    cycling_cb_neg = 0

    add_agent!(
        TransportAgent, model, age, gender, education, employment, income,
        original_transport_type, transport_type, transport_choice,
        av_attitudes, av_social_norms, av_control_factors, av_behavioural_intention, av_subjective_norm, av_facilitating_conditions, av_threshold,
        rh_attitudes, rh_social_norms, rh_control_factors, rh_behavioural_intention, rh_subjective_norm, rh_facilitating_conditions, rh_threshold, rh_fee_applied,
        near_public_transport, impulsivity, av_cb_pos, av_cb_neg, rh_cb_pos, rh_cb_neg,
        physical_health_layer, sedentary_behaviour,
        is_pt_agent, pt_coverage_radius, fee, is_promotion_agent, promotion_coverage_radius, av_advertising_efficacy, pt_active_transport_advertising_efficacy,
        walking_attitudes, walking_social_norms, walking_control_factors, walking_behavioural_intention, walking_subjective_norm, walking_facilitating_conditions, walking_threshold,
        cycling_attitudes, cycling_social_norms, cycling_control_factors, cycling_behavioural_intention, cycling_subjective_norm, cycling_facilitating_conditions, cycling_threshold,
        walking_cb_pos, walking_cb_neg, cycling_cb_pos, cycling_cb_neg
        )
    end 


    return model 
end



model = initialize()



# IF USING A SIGMOID FUNCTION 

# function sigmoid(x)
#     return 1 / (1 + exp(-x))
# end

# function agent_health!(agent, model)
#     if agent.transport_choice == 1 || agent.transport_choice == 2 || agent.transport_choice == 6
#         if agent.original_transport_type != agent.transport_type == 1
#             increment = 0.1 * sigmoid(agent.sedentary_behaviour - 5)  # adjust the '5' based on your specific scenario
#             agent.sedentary_behaviour += increment
#             agent.physical_health_layer -= 0.01  # Adjust this value based on how you'd like health to decay
#         end
#     end
# end

# EXPONENTIAL FUNCTION

# function exponential_increase(x, a=0.1, b=0.2)
#     return a * exp(b * x)
# end

# function agent_health!(agent, model)
#     # Check if original transport type was either cycling or walking
#     if agent.original_transport_type == 4 || agent.original_transport_type == 5
#         # Check if current transport choice is AV, Car, or RideHailApp
#         if agent.transport_choice == 1 || agent.transport_choice == 2 || agent.transport_choice == 6
#             # Increase sedentary behavior based on the exponential function
#             increment = exponential_increase(agent.sedentary_behaviour)
#             agent.sedentary_behaviour += increment
#             agent.physical_health_layer -= 0.01  # Adjust this value for health decay if necessary
#         end
#     end
# end


# POLICY: 10% REBATE ON THE PURCHASE PRICE 
# Note: the results from this do not change anything, as it does not change the perceived purchase price. Option 2 will address this. 

const AV_REBATE = 0.1 # 10% rebate for using AV

function apply_rebate_after_purchase!(agent, model)
    if agent.transport_choice == 1 # If choice is AV
        rebate_amount = model.private_AV_cost * AV_REBATE
        agent.income += rebate_amount # Apply the rebate to the income
    end
end



adata = [:pos, :transport_choice, :age, :income, :original_transport_type, :av_attitudes, :av_social_norms, :av_control_factors, :av_facilitating_conditions, :av_subjective_norm, :rh_attitudes, :rh_social_norms, :rh_control_factors, :rh_behavioural_intention, :impulsivity, :physical_health_layer, :sedentary_behaviour]
model = initialize()
adf, mdf = run!(model, 120; adata)

CSV.write("C:/Users/godicb/OneDrive - The University of Melbourne/Documents/Julia/AV_Transport_ABM/output20_08042024.csv" ,agent_df)

using CairoMakie # using a different plotting backend that enables interactive plots
using Statistics




# GRAPH NUMBER 1: TRANSPORT CHOICES 

av_user(a) = (a.transport_choice == 1)
auto_rh_user(a) = (a.transport_choice == 9)
car_user(a) = (a.transport_choice == 2)
pt_user(a) = (a.transport_choice == 3)
personal_micromobility_user(a) = (a.transport_choice == 4)
walker(a) = (a.transport_choice == 6)
cyclist(a) = (a.transport_choice == 5) 
rh_user(a) = (a.transport_choice == 7) 
carsharing_user(a) = (a.transport_choice == 8) 

print(model.AVs_time_series)
print(model.RH_trips_time_series)


avcount(model) = sum(model.AVs_time_series)
rhcount(model) = sum(model.RH_trips_time_series)


steps = 500
adata = [(av_user, count), (auto_rh_user, count), (rh_user, count), (car_user, count), (pt_user, count), (personal_micromobility_user, count), (walker, count), (cyclist, count), (carsharing_user, count)]
mdata = [avcount, rhcount]

agent_df, model_df = run!(model, 150; adata = adata, mdata = mdata)



function plot_population_timeseries(agent_df)
    figure = Figure(resolution = (1200, 400))
    ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population")
    av_agents = lines!(ax, agent_df.time, agent_df.count_av_user, color = :blue)
    auto_rh_agents = lines!(ax, agent_df.time, agent_df.count_auto_rh_user, color = :green)
    car_agents = lines!(ax, agent_df.time, agent_df.count_car_user, color = :purple)
    pt_agents = lines!(ax, agent_df.time, agent_df.count_pt_user, color = :orange)
    personal_micromobility_agents = lines!(ax, agent_df.time, agent_df.count_personal_micromobility_user, color = :red)
    walker_agents = lines!(ax, agent_df.time, agent_df.count_walker, color = :pink)
    cyclist_agents = lines!(ax, agent_df.time, agent_df.count_cyclist, color = :yellow)
    rh_agents = lines!(ax, agent_df.time, agent_df.count_rh_user, color = :magenta)
    carsharing_agents = lines!(ax, agent_df.time, agent_df.count_carsharing_user, color = :cyan)
    # av_population = lines!(ax, mdf.step, mdf.avcount, color = :green)
    # rh_population = lines!(ax, mdf.step, mdf.rhcount, color = :blue)
    figure[1, 2] = Legend(figure, [av_agents, auto_rh_agents, car_agents, pt_agents, personal_micromobility_agents, walker_agents, cyclist_agents, rh_agents, carsharing_agents], ["AVs", "Autonomous RH users", "Car users", "Public transport", "Personal micromobility users", "Walkers", "Cyclists", "Ride hail users", "Car sharing users"])
    figure
end




# Plot first graph: 
plot_population_timeseries(agent_df)

# GRAPH NUMBER 2: PLOT HEALTH OUTCOMES 

# using DataFrames

# travel_agents(a) = (a.is_promotion_agent == false) && (a.is_pt_agent == false)
# sedentary_behaviour_plot(travel_agents) = (travel_agents.sedentary_behaviour) 


# sedentary(model) = sum(model.AVs_time_series)
# rhcount(model) = sum(model.RH_trips_time_series)

# steps = 500
# adata = [(sedentary_behaviour_plot, mean)]
# mdata = [avcount, rhcount]


# adf, mdf = run!(model, agent_step!, model_step!, steps; adata, mdata)

# function plot_population_health(adf)
#     figure = Figure(resolution = (600, 400))
#     ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Sedentary behaviour")
#     sedentary_agent_plot_1 = lines!(ax, adf.step, adf.mean_sedentary_behaviour_plot, color = :blue)
#     figure[1, 2] = Legend(figure, [sedentary_agent_plot_1], ["Sedentary behaviour"])
#     figure
# end

# CSV.write("C:/Users/godicb/OneDrive - The University of Melbourne/Documents/Julia/AV_Transport_ABM/output1_23022024.csv" ,adf)

# plot_population_health(adf)






# OLD DECISION FUNCTIONS 


# function av_decision!(agent, model)
#     attitudes = agent.av_attitudes
#     control_behaviour = agent.av_cb_pos - 2*agent.av_cb_neg
#     # calculating social_norms
#     subjective_norms = 0
#     num_neighbors = 0
#     for neighbor in nearby_agents(agent, model)
#         subjective_norms = subjective_norms + neighbor.av_attitudes
#         num_neighbors = num_neighbors + 1
#     end
    
#     #Taking the average
#     subjective_norms = num_neighbors==0 ? 0 : subjective_norms / num_neighbors

#     #Taking the average
#     descriptive_norms = model.AVs / model.total_agents


#     # REBATE POLICY OPTION 2:
#     # If a rebate policy with the full amount is being applied, included, ensure that the below code is activated:
#     apply_rebate!(agent, model, AV_rebate_full_amount)

#     facil_conditions = [agent.income > model.private_AV_cost]

#     AV_decision = AV_TPB(attitudes, control_behaviour, subjective_norms, descriptive_norms, facil_conditions, model.av_threshold_model)
    
#     if AV_decision 
#         push!(model.AVs_time_series, model.AVs_time_series[end] + 1)
#     else
#         push!(model.AVs_time_series, model.AVs_time_series[end])
#     end

#     if AV_decision
#         agent.transport_choice = 1
#     end

# end



# function rh_decision!(agent, model)
#     attitudes = agent.rh_attitudes
#     control_behaviour = agent.rh_cb_pos - 2*agent.rh_cb_neg
#     # calculating social_norms
#     subjective_norms = 0
#     num_neighbors = 0
#     for neighbor in nearby_agents(agent, model)
#         subjective_norms = subjective_norms + neighbor.rh_attitudes
#         num_neighbors = num_neighbors + 1
#     end
    
#     # Taking the average
#     subjective_norms = num_neighbors == 0 ? 0 : subjective_norms / num_neighbors

#     # Taking the average
#     descriptive_norms = model.RH_trips / model.total_agents
    
#     agent_trip_distance = rand(1:20)

#     print("Trip distance: $agent_trip_distance")

#     # SHORT TRIP FEE POLICY IMPLEMENTED - ACTIVATE CODE BELOW #

#     if agent_trip_distance <= 5
#         facil_conditions = [(agent.income*0.001) > assign_rh_trip_cost(agent_trip_distance, agent, model)]
#     else
#         facil_conditions = [(agent.income*0.001) > model.rh_trip_cost]
#     end

#     # SHORT TRIP FEE POLICY NOT IMPLEMENTED - ACTIVATE CODE BELOW #

#     #facil_conditions = [(agent.income*0.001) > model.rh_trip_cost]

#     RH_decision = RH_TPB(attitudes, control_behaviour, subjective_norms, descriptive_norms, facil_conditions, model.rh_threshold_model)

#     if RH_decision 
#         push!(model.RH_trips_time_series, model.RH_trips_time_series[end] + 1)
#     else
#         push!(model.RH_trips_time_series, model.RH_trips_time_series[end])
#     end

#     if RH_decision
#         agent.transport_choice = 6
#     end

# end


# function walking_TBP(walking_attitudes, walking_control_factors, walking_social_norms, walking_subjective_norm, walking_facilitating_conditions, walking_threshold)
#     if false
#         print("Facil Conditions: ", walking_facilitating_conditions, " Sum: ", sum(walking_attitudes)+sum(walking_control_factors)+sum(walking_social_norms)+sum(walking_subjective_norm),
#             " Threshold: ", walking_threshold, "\n")
#     end 
#     return all(walking_facilitating_conditions) &&
#         sum(walking_attitudes)+
#         sum(walking_control_factors)+
#         sum(walking_social_norms)+
#         sum(walking_subjective_norm) > walking_threshold
#     print(return)
# end

# function cycling_TBP(cycling_attitudes, cycling_control_factors, cycling_social_norms, cycling_subjective_norm, cycling_facilitating_conditions, cycling_threshold)
#     if false
#         print("Facil Conditions: ", cycling_facilitating_conditions, " Sum: ", sum(cycling_attitudes)+sum(cycling_control_factors)+sum(cycling_social_norms)+sum(cycling_subjective_norm),
#             " Threshold: ", cycling_threshold, "\n")
#     end 
#     return all(cycling_facilitating_conditions) &&
#         sum(cycling_attitudes)+
#         sum(cycling_control_factors)+
#         sum(cycling_social_norms)+
#         sum(cycling_subjective_norm) > cycling_threshold
#     print(return)
# end








