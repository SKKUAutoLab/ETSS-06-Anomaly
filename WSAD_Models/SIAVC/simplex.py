class Simplex(object):
	def __init__(self, tableau ):
		super(Simplex, self).__init__()
		self.tableau = tableau

	def phase1(self):
		res = True
		if self.checkFeasibility() == False:
			self.addArtificialVariables()
			self.addNewCostFunction()
			simplex = Simplex(self.tableau)
			count = 1
			while simplex.canContinue():
				simplex.iteration()
				count += 1
			if len(set(self.tableau.basis) & set(self.tableau.artificial_variable)) > 0:
				res =  False
			else:
				if self.tableau[self.tableau.cost_index][self.tableau.b_index] > 0:
					self.solution = 'infeasible'
					res = False
				else:
					self.tableau.removeRow(self.tableau.cost_index)
					self.tableau.cost_index = self.tableau.lines -1
					for i in self.tableau.artificial_variable:
						self.tableau.removeColumn(self.tableau.columns-2)
					self.tableau.b_index = self.tableau.columns-1
		return res

	def phase2(self):
		i = 1
		b = True
		while self.canContinue():
			b = self.iteration()
			if b == False:
				break
			i += 1
		if b == True:
			self.solution = self.tableau[self.tableau.cost_index][self.tableau.b_index]

	def execute(self):
		self.solution = None
		r = self.phase1()
		if r == True:
			self.phase2()

	def requeredArtificalVariables(self):
		n = self.tableau.constraints_count
		n_of_variables = 0
		c = list()
		for i in range(0,n):
			for j in range(self.tableau.var_count,self.tableau.columns):
				if j - self.tableau.var_count == i:
					if self.tableau[i][j] != 1:
						n_of_variables += 1
						c.append(i)
		return n_of_variables,c

	def addArtificialVariables(self):
		n_of_variables,c = self.requeredArtificalVariables()
		if n_of_variables  > 0:
			self.tableau.artificial_variable_count = n_of_variables
			for i in range(0,n_of_variables):
				idx = self.tableau.columns - 1
				self.tableau.addColumn(idx,0.0)
			idx = self.tableau.var_count+self.tableau.constraints_count
			for r in c:
				self.tableau.artificial_variable.append(idx)
				self.tableau[r][idx] = 1.0
				self.tableau.basis.remove(r+self.tableau.var_count)
				self.tableau.basis.append(idx)
				idx += 1
		self.tableau.b_index = self.tableau.columns - 1

	def addNewCostFunction(self):
		n_of_variables,c = self.requeredArtificalVariables()
		if n_of_variables > 0:
			self.tableau.addRow(self.tableau.lines,0.0)
			idx = self.tableau.lines - 1
			for a in self.tableau.artificial_variable:
				self.tableau[idx][a] = 1.0
			for r in c:
				for i in range(0,self.tableau.columns):
					self.tableau[idx][i] = self.tableau[idx][i] -  self.tableau[r][i]
			self.tableau.cost_index = self.tableau.lines - 1

	def canContinue(self):
		cost_index = self.tableau.cost_index
		for i in range (0,self.tableau.columns):
			if self.tableau[cost_index][i] < 0:
				return True
		return False

	def checkFeasibility(self):
		n,c = self.requeredArtificalVariables()
		if n == 0:
			return True
		return False

	def getPivot(self):
		cost_index = self.tableau.cost_index
		pivot = 0
		for i in range (0,self.tableau.columns-1):
			if self.tableau[cost_index][i] < pivot:
				pivot = self.tableau[cost_index][i]

		return self.tableau[cost_index].index(pivot)

	def isBoundedSolution(self,pivot):
		for i in range(0,self.tableau.constraints_count):
			if self.tableau[i][pivot] > 0 :
				return True
		return False

	def isDegenerative(self,pivot):
		b_index = self.tableau.b_index
		limit_set  = set()
		for i in range(0,self.tableau.constraints_count):
			if self.tableau[i][pivot] > 0 :
				limit = self.tableau[i][b_index]/self.tableau[i][pivot]
				if limit in limit_set:
					return True
		return False

	def getConstraintLimit(self,pivot):
		b_index = self.tableau.b_index
		limit = float("inf")
		line_index = -1
		for i in range(0,self.tableau.constraints_count):
			if self.tableau[i][pivot] > 0:
				if self.tableau[i][b_index]/self.tableau[i][pivot] < limit:
					limit = self.tableau[i][b_index]/self.tableau[i][pivot]
					line_index = i
		return line_index

	def scalingMatrix(self,i,pivot):
		for j in range(0,self.tableau.lines):
			if i != j:
				pivot_value = self.tableau[j][pivot]
				for k in range(0,self.tableau.columns):
					self.tableau[j][k] = self.tableau[j][k] - (self.tableau[i][k]*pivot_value)

	def gaussianOperation(self,constraint_index,pivot_index):
		pivot_value = self.tableau[constraint_index][pivot_index]
		for i in range(0,self.tableau.columns):
			self.tableau[constraint_index][i] = self.tableau[constraint_index][i] / pivot_value
		self.scalingMatrix(constraint_index,pivot_index)

	def iteration(self):
		pivot_index  = self.getPivot()
		if(self.isDegenerative(pivot_index)):
			self.solution = 'degenerative'
			return False
		if(self.isBoundedSolution(pivot_index)):
			constraint_index = self.getConstraintLimit(pivot_index)
			if constraint_index == -1:
				return
			self.tableau.changeBasis(pivot_index,constraint_index)
			self.gaussianOperation(constraint_index,pivot_index)
			return True
		else:
			self.solution = 'unbounded'
			return False