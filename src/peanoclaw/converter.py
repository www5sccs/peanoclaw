
def get_number_of_dimensions( q ):
    """ This function is used to find out the dimension of problem with the help of the q array
    since the solution field have one element more which contains the number of unknowns per subcell the dimension 
    computed by substracting one from the size of the solution list.
    """
    size = len(q)
    return size - 1

if __name__ == '__main__':
    pass

def create_domain(number_of_dimensions, position, size, subdivision_factor):
  """ Creates a domain from the information given by callbacks.
  """
  from clawpack.pyclaw import Dimension
  from clawpack.pyclaw import Domain
  
  if number_of_dimensions is 2:
    dim_x = Dimension('x', position[0], position[0] + size[0], subdivision_factor[0])
    dim_y = Dimension('y', position[1], position[1] + size[1], subdivision_factor[1])
    return Domain([dim_x, dim_y])
  elif number_of_dimensions is 3:
    dim_x = Dimension('x', position[0], position[0] + size[0], subdivision_factor[0])
    dim_y = Dimension('y', position[1], position[1] + size[1], subdivision_factor[1])
    dim_z = Dimension('z', position[2], position[2] + size[2], subdivision_factor[2])
    return Domain([dim_x, dim_y, dim_z])
    
def create_subgrid_state(global_state, domain, q, qbc, aux, unknowns_per_cell, aux_fields_per_cell):
  from clawpack.pyclaw import State
  subgrid_state = State(domain, unknowns_per_cell, aux_fields_per_cell)
  subgrid_state.q = q
  subgrid_state.aux = aux
  subgrid_state.problem_data = global_state.problem_data
  
  return subgrid_state
