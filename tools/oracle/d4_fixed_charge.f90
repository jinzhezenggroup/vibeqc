! Test-only adapter to independent upstream DFT-D4. No VibeQC mathematics.
! SPDX-License-Identifier: GPL-3.0-or-later
program d4_fixed_charge_oracle
  use mctc_env, only: wp, error_type
  use mctc_io, only: structure_type, new
  use dftd4_model, only: d4_model, new_d4_model, d4_qmod
  use dftd4_damping_rational, only: rational_damping_param
  use dftd4_ncoord, only: get_coordination_number, add_coordination_number_derivs
  implicit none
  type(structure_type) :: mol
  type(d4_model) :: model
  type(rational_damping_param) :: param
  type(error_type), allocatable :: error
  integer :: n, i, nr
  integer, allocatable :: numbers(:)
  real(wp), allocatable :: xyz(:,:), q(:), cn(:), energies(:), dc(:), dq(:), qgrad(:)
  real(wp), allocatable :: weights(:,:,:), wc(:,:,:), wq(:,:,:), c6(:,:), cc(:,:), cq(:,:)
  real(wp), allocatable :: grad(:,:)
  real(wp) :: trans(3,1), sigma(3,3), e2
  read(*,*) n
  if(n<1 .or. n>256) error stop 'unsupported oracle size'
  read(*,*) param%s6, param%s8, param%s9, param%a1, param%a2
  allocate(numbers(n),xyz(3,n),q(n),cn(n),energies(n),dc(n),dq(n),qgrad(n),grad(3,n))
  do i=1,n
    read(*,*) numbers(i),xyz(:,i),q(i)
  end do
  call new(mol,numbers,xyz)
  call new_d4_model(error,model,mol,qmod=d4_qmod%gfn2)
  if(allocated(error)) error stop 'D4 model construction failed'
  trans=0.0_wp
  call get_coordination_number(mol,trans,30.0_wp,model%rcov,model%en,cn)
  nr=maxval(model%ref)
  allocate(weights(nr,n,model%ncoup),wc(nr,n,model%ncoup),wq(nr,n,model%ncoup))
  allocate(c6(n,n),cc(n,n),cq(n,n))
  call model%weight_references(mol,cn,q,weights,wc,wq)
  call model%get_atomic_c6(mol,weights,wc,wq,c6,cc,cq)
  energies=0.0_wp
  dc=0.0_wp
  dq=0.0_wp
  grad=0.0_wp
  sigma=0.0_wp
  call param%get_dispersion2(mol,trans,50.0_wp,0.0_wp,model%r4r2,c6,cc,cq,energies,dc,dq,grad,sigma)
  e2=sum(energies)
  qgrad=dq
  q=0.0_wp
  call model%weight_references(mol,cn,q,weights,wc,wq)
  call model%get_atomic_c6(mol,weights,wc,wq,c6,cc,cq)
  call param%get_dispersion3(mol,trans,25.0_wp,0.0_wp,model%r4r2,c6,cc,cq,energies,dc,dq,grad,sigma)
  call add_coordination_number_derivs(mol,trans,30.0_wp,model%rcov,model%en,dc,grad,sigma)
  write(*,'(*(es25.17e3,1x))') e2,sum(energies)-e2
  write(*,'(*(es25.17e3,1x))') grad
  write(*,'(*(es25.17e3,1x))') qgrad
end program d4_fixed_charge_oracle
